"""Celebrity audio data pipeline.

Downloads short clips from YouTube (yt-dlp + ffmpeg), preprocesses them into
fixed [1, 40, 125] MFCC tensors, and builds the forget/retain scenario split
used by celebrity_benchmark.py.

Directory layout produced
-------------------------
data/celebrity/
    trump/     *.wav  (16 kHz mono)
    biden/     *.wav
    obama/     *.wav
    harris/    *.wav
    sanders/   *.wav
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import torchaudio
    import torchaudio.transforms as T
except ImportError as exc:
    raise SystemExit("torchaudio is required. pip install torchaudio") from exc


# ---------------------------------------------------------------------------
# Constants (shared with benchmark.py)
# ---------------------------------------------------------------------------

TARGET_SR = 16_000
CLIP_NUM_SAMPLES = 32_000   # 2 s at 16 kHz
N_MFCC = 40
MFCC_TIME_STEPS = 125

FORGET_CELEBRITY = "trump"
RETAIN_CELEBRITIES = ["biden", "obama", "harris", "sanders"]
ALL_CELEBRITIES = [FORGET_CELEBRITY] + RETAIN_CELEBRITIES

CELEBRITY_LABEL: Dict[str, int] = {name: idx for idx, name in enumerate(ALL_CELEBRITIES)}

# ---------------------------------------------------------------------------
# Manifest
# Each entry: (youtube_url, start_time, end_time)  — times as "MM:SS" strings
# These are illustrative placeholders; fill in real URLs before running.
# ---------------------------------------------------------------------------

CELEBRITY_MANIFEST: Dict[str, List[Tuple[str, str, str]]] = {
    "trump": [
        # Example: ("https://www.youtube.com/watch?v=XXXXXXXXXXX", "0:10", "0:40"),
    ],
    "biden": [],
    "obama": [],
    "harris": [],
    "sanders": [],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CelebRecord:
    """One preprocessed utterance from a celebrity speaker."""

    sample_id: str
    celebrity_name: str
    label: int
    features: torch.Tensor  # [1, 40, 125]


@dataclass
class ScenarioSplit:
    """Forget / retain split for the celebrity unlearning scenario."""

    name: str
    description: str
    forget_train: List[CelebRecord]
    forget_test: List[CelebRecord]
    retain_train: List[CelebRecord]
    retain_test: List[CelebRecord]
    mia_holdout: List[CelebRecord]

    @property
    def original_train(self) -> List[CelebRecord]:
        return [*self.retain_train, *self.forget_train]

    @property
    def full_test(self) -> List[CelebRecord]:
        return [*self.retain_test, *self.forget_test]


class RecordDataset(Dataset):
    """Wrap CelebRecord list as a DataLoader-compatible Dataset."""

    def __init__(self, records: Sequence[CelebRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        r = self.records[index]
        return r.features, r.label


# ---------------------------------------------------------------------------
# MFCC helpers
# ---------------------------------------------------------------------------

def _build_mfcc_transform() -> T.MFCC:
    return T.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 512, "hop_length": 256, "n_mels": 64},
    )


def _preprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    mfcc_transform: T.MFCC,
    resamplers: Dict[int, T.Resample],
) -> torch.Tensor:
    """Identical pipeline to benchmark.py — mono, fixed length, [1, 40, 125]."""
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SR:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = T.Resample(sample_rate, TARGET_SR)
        waveform = resamplers[sample_rate](waveform)

    if waveform.size(1) < CLIP_NUM_SAMPLES:
        waveform = F.pad(waveform, (0, CLIP_NUM_SAMPLES - waveform.size(1)))
    else:
        waveform = waveform[:, :CLIP_NUM_SAMPLES]

    mfcc = mfcc_transform(waveform).float()
    if mfcc.size(-1) < MFCC_TIME_STEPS:
        mfcc = F.pad(mfcc, (0, MFCC_TIME_STEPS - mfcc.size(-1)))
    else:
        mfcc = mfcc[..., :MFCC_TIME_STEPS]

    return mfcc


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_celebrity_clips(
    name: str,
    manifest_entries: List[Tuple[str, str, str]],
    out_dir: Path,
    max_clips: int = 50,
) -> None:
    """Download and segment audio clips for one celebrity via yt-dlp + ffmpeg.

    Each manifest entry is (youtube_url, start_time, end_time).
    Clips are saved as <out_dir>/<name>_<idx>.wav at 16 kHz mono.

    Skips already-downloaded clips so reruns are idempotent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_idx = 0

    for url, start, end in manifest_entries:
        if clip_idx >= max_clips:
            break

        out_path = out_dir / f"{name}_{clip_idx:04d}.wav"
        if out_path.exists():
            clip_idx += 1
            continue

        # Download full video audio to a temp file
        tmp_path = out_dir / f"_tmp_{name}_{clip_idx:04d}.%(ext)s"
        dl_cmd = [
            "yt-dlp",
            "--quiet",
            "--no-playlist",
            "-x",
            "--audio-format", "wav",
            "-o", str(tmp_path),
            url,
        ]
        try:
            subprocess.run(dl_cmd, check=True, timeout=120)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[download] WARN: failed to download {url}: {exc}")
            continue

        # Find the downloaded file
        tmp_files = list(out_dir.glob(f"_tmp_{name}_{clip_idx:04d}.*"))
        if not tmp_files:
            print(f"[download] WARN: no file found after download for {url}")
            continue
        tmp_file = tmp_files[0]

        # Trim and resample with ffmpeg
        trim_cmd = [
            "ffmpeg", "-y",
            "-ss", start,
            "-to", end,
            "-i", str(tmp_file),
            "-ar", "16000",
            "-ac", "1",
            str(out_path),
        ]
        try:
            subprocess.run(trim_cmd, check=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[download] WARN: ffmpeg trim failed for {url}: {exc}")
        finally:
            tmp_file.unlink(missing_ok=True)

        if out_path.exists():
            clip_idx += 1
            print(f"[download] {name}: saved {out_path.name}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_celebrity_records(
    data_dir: Path,
    speakers: Optional[List[str]] = None,
    samples_per_speaker: int = 50,
) -> Dict[str, List[CelebRecord]]:
    """Walk data_dir/<name>/*.wav and return CelebRecord lists keyed by name.

    Args:
        data_dir:             Root of celebrity clips (e.g. data/celebrity/).
        speakers:             Subset of speakers to load; defaults to ALL_CELEBRITIES.
        samples_per_speaker:  Maximum clips to load per speaker.

    Returns:
        Dict mapping celebrity name -> list of CelebRecord.
    """
    if speakers is None:
        speakers = ALL_CELEBRITIES

    mfcc_transform = _build_mfcc_transform()
    resamplers: Dict[int, T.Resample] = {}
    records: Dict[str, List[CelebRecord]] = {}

    for name in speakers:
        speaker_dir = data_dir / name
        wav_files = sorted(speaker_dir.glob("*.wav"))[:samples_per_speaker]

        if not wav_files:
            print(f"[load] WARN: no wav files found in {speaker_dir}")
            records[name] = []
            continue

        label = CELEBRITY_LABEL[name]
        speaker_records: List[CelebRecord] = []

        for wav_path in wav_files:
            try:
                waveform, sr = torchaudio.load(str(wav_path))
            except Exception as exc:
                print(f"[load] WARN: could not load {wav_path}: {exc}")
                continue

            features = _preprocess_waveform(waveform, sr, mfcc_transform, resamplers)
            speaker_records.append(
                CelebRecord(
                    sample_id=wav_path.stem,
                    celebrity_name=name,
                    label=label,
                    features=features,
                )
            )

        records[name] = speaker_records
        print(f"[load] {name}: {len(speaker_records)} clips loaded")

    return records


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_celebrity_scenario(
    records: Dict[str, List[CelebRecord]],
    forget_speaker: str = FORGET_CELEBRITY,
    forget_test_fraction: float = 0.20,
    retain_test_fraction: float = 0.20,
    mia_holdout_fraction: float = 0.20,
    seed: int = 42,
) -> ScenarioSplit:
    """Build the forget/retain train/test split for the celebrity scenario.

    The forget speaker's clips are split into forget_train / forget_test / mia_holdout.
    Each retain speaker's clips are split into retain_train / retain_test.
    """
    rng = random.Random(seed)

    forget_records = list(records.get(forget_speaker, []))
    rng.shuffle(forget_records)

    n_forget = len(forget_records)
    n_ft = max(1, int(n_forget * forget_test_fraction))
    n_mia = max(1, int(n_forget * mia_holdout_fraction))
    forget_test = forget_records[:n_ft]
    mia_holdout = forget_records[n_ft: n_ft + n_mia]
    forget_train = forget_records[n_ft + n_mia:]

    retain_train: List[CelebRecord] = []
    retain_test: List[CelebRecord] = []

    for name, recs in records.items():
        if name == forget_speaker:
            continue
        recs_copy = list(recs)
        rng.shuffle(recs_copy)
        n_test = max(1, int(len(recs_copy) * retain_test_fraction))
        retain_test.extend(recs_copy[:n_test])
        retain_train.extend(recs_copy[n_test:])

    return ScenarioSplit(
        name="celebrity_scenario1",
        description=f"Forget {forget_speaker}; retain {RETAIN_CELEBRITIES}",
        forget_train=forget_train,
        forget_test=forget_test,
        retain_train=retain_train,
        retain_test=retain_test,
        mia_holdout=mia_holdout,
    )
