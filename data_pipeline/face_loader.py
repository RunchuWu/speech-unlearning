"""Face-frame helpers for the multimodal celebrity benchmark.

This module samples a small number of RGB video frames, resizes them to a
fixed resolution, and mean-pools them into one `[3, 96, 96]` tensor per clip.
It intentionally keeps the pipeline lightweight so it runs comfortably on a
MacBook without adding heavy face-detection dependencies.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .celebrity_loader import CelebRecord

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


@dataclass(frozen=True)
class MultimodalRecord:
    """One aligned audio-plus-video sample for speaker identification."""

    sample_id: str
    celebrity_name: str
    label: int
    audio_features: torch.Tensor  # [1, 40, 125]
    face_features: torch.Tensor   # [3, 96, 96]


def _probe_duration_seconds(video_path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None

    return duration if duration > 0 else None


def _frame_timestamps(duration_seconds: float, num_frames: int) -> List[float]:
    step = duration_seconds / max(num_frames + 1, 1)
    return [step * (idx + 1) for idx in range(num_frames)]


def _extract_frame(video_path: Path, timestamp_seconds: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{timestamp_seconds:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        str(out_path),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )


def _load_image_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def extract_face_frames(
    video_path: Path | str,
    num_frames: int = 5,
    image_size: int = 96,
) -> torch.Tensor:
    """Extract uniformly spaced RGB frames and mean-pool them to `[3, H, W]`."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = _probe_duration_seconds(video_path)
    if duration is None:
        raise RuntimeError(f"Could not read duration for video: {video_path}")

    frame_tensors: List[torch.Tensor] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, timestamp in enumerate(_frame_timestamps(duration, num_frames)):
            frame_path = tmp_root / f"frame_{idx:02d}.png"
            try:
                _extract_frame(video_path, timestamp, frame_path)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
            if frame_path.exists():
                frame_tensors.append(_load_image_tensor(frame_path, image_size))

    if not frame_tensors:
        raise RuntimeError(f"No frames could be extracted from video: {video_path}")

    return torch.stack(frame_tensors, dim=0).mean(dim=0)


def _find_video_for_sample(sample_id: str, video_dir: Path) -> Optional[Path]:
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate

    for candidate in sorted(video_dir.glob(f"{sample_id}.*")):
        if candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate

    return None


def load_multimodal_records(
    audio_records_by_speaker: Dict[str, Sequence[CelebRecord]],
    video_root: Path,
    num_frames: int = 5,
    image_size: int = 96,
) -> Dict[str, List[MultimodalRecord]]:
    """Align audio records with companion videos and return multimodal samples."""
    multimodal_records: Dict[str, List[MultimodalRecord]] = {}

    for celebrity_name, audio_records in audio_records_by_speaker.items():
        video_dir = video_root / celebrity_name
        speaker_records: List[MultimodalRecord] = []

        if not video_dir.exists():
            print(f"[face] WARN: video directory not found for {celebrity_name}: {video_dir}")
            multimodal_records[celebrity_name] = speaker_records
            continue

        for record in audio_records:
            video_path = _find_video_for_sample(record.sample_id, video_dir)
            if video_path is None:
                print(f"[face] WARN: no companion video for sample {record.sample_id}")
                continue

            try:
                face_features = extract_face_frames(
                    video_path,
                    num_frames=num_frames,
                    image_size=image_size,
                )
            except Exception as exc:
                print(f"[face] WARN: failed to extract frames for {video_path}: {exc}")
                continue

            speaker_records.append(
                MultimodalRecord(
                    sample_id=record.sample_id,
                    celebrity_name=record.celebrity_name,
                    label=record.label,
                    audio_features=record.features,
                    face_features=face_features,
                )
            )

        multimodal_records[celebrity_name] = speaker_records
        print(f"[face] {celebrity_name}: {len(speaker_records)} multimodal clips loaded")

    return multimodal_records
