"""Minimal speech unlearning benchmark on LibriSpeech speaker classification.

This script builds a small but rigorous benchmark that:
1. Loads five LibriSpeech speakers and extracts fixed-size MFCC features.
2. Trains an original speaker classifier and a retrain oracle from scratch.
3. Applies two unlearning methods with a shared interface.
4. Evaluates forgetting, retention, membership inference, and overall utility.

Scenario 1 is run by default:
- Df = all benchmark utterances from speaker 1089
- Dr = all benchmark utterances from the remaining speakers

Scenario 2 is implemented but not executed by default:
- Df = random 10% of benchmark utterances across all speakers
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
    import torchaudio.transforms as T
except ImportError as exc:
    raise SystemExit(
        "torchaudio is required for benchmark.py. "
        f"Install a version that matches torch=={torch.__version__}."
    ) from exc


SEED = 42
REPO_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = REPO_ROOT / "artifacts"
BENCHMARK_ARTIFACT_DIR = ARTIFACT_ROOT / "benchmark"
DATA_ROOT = REPO_ROOT / "data"
LIBRISPEECH_SPLIT = "test-clean"
SPEAKER_IDS = [1089, 2094, 3570, 4077, 5142]
FORGET_SPEAKER_ID = 1089
SAMPLES_PER_SPEAKER = 50
ALLOW_SHORT_SPEAKER_FALLBACK = True

SCENARIO_TO_RUN = 1
SAMPLE_SCENARIO_FORGET_FRACTION = 0.10
FORGET_TEST_FRACTION = 0.20
RETAIN_TEST_FRACTION = 0.20
MIA_HOLDOUT_FRACTION = 0.20

TARGET_SR = 16_000
CLIP_NUM_SAMPLES = 32_000
N_MFCC = 40
MFCC_TIME_STEPS = 125
BATCH_SIZE = 16
NUM_WORKERS = 0

TRAIN_EPOCHS = 60
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_STEP_SIZE = 20
TRAIN_GAMMA = 0.5

GA_EPOCHS = 20
GA_LR = 1e-4

RL_EPOCHS = 20
RL_LR = 5e-5

ORIGINAL_CHECKPOINT = BENCHMARK_ARTIFACT_DIR / "model_original.pt"
RETRAIN_CHECKPOINT = BENCHMARK_ARTIFACT_DIR / "model_retrain.pt"
RESULTS_FIGURE = BENCHMARK_ARTIFACT_DIR / "results.png"
RESULTS_CSV = BENCHMARK_ARTIFACT_DIR / "results_summary.csv"
GA_STEPWISE_FIGURE = BENCHMARK_ARTIFACT_DIR / "ga_stepwise.png"
GA_STEPWISE_CSV = BENCHMARK_ARTIFACT_DIR / "ga_stepwise.csv"
RL_STEPWISE_FIGURE = BENCHMARK_ARTIFACT_DIR / "rl_stepwise.png"
RL_STEPWISE_CSV = BENCHMARK_ARTIFACT_DIR / "rl_stepwise.csv"

METHOD_NAMES = [
    "No Unlearning",
    "Retrain Oracle",
    "Gradient Ascent",
    "Random Label",
]


@dataclass(frozen=True)
class AudioRecord:
    """Store one preprocessed utterance so every method sees identical fixed features."""

    sample_id: str
    speaker_id: int
    label: int
    features: torch.Tensor


@dataclass
class ScenarioSplit:
    """Bundle one benchmark split so training, testing, and MIA use disjoint partitions."""

    name: str
    description: str
    forget_train: List[AudioRecord]
    forget_test: List[AudioRecord]
    retain_train: List[AudioRecord]
    retain_test: List[AudioRecord]
    mia_holdout: List[AudioRecord]

    @property
    def original_train(self) -> List[AudioRecord]:
        return [*self.retain_train, *self.forget_train]

    @property
    def full_test(self) -> List[AudioRecord]:
        return [*self.retain_test, *self.forget_test]


class RecordDataset(Dataset):
    """Wrap in-memory MFCC tensors so DataLoader can batch them without re-reading audio."""

    def __init__(self, records: Sequence[AudioRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        return record.features, record.label


class TinySpeakerCNN(nn.Module):
    """Tiny 3-block CNN that matches the requested [B, 1, 40, 125] MFCC input shape."""

    def __init__(self, num_classes: int = len(SPEAKER_IDS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.pool(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    """Fix random seeds so the demo is reproducible across training and unlearning runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    """Prefer MPS when its pooling ops work; otherwise use CPU so the benchmark still runs."""

    if torch.backends.mps.is_available():
        try:
            probe = torch.randn(1, 1, N_MFCC, MFCC_TIME_STEPS, device="mps")
            layer = nn.Sequential(
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            ).to("mps")
            with torch.no_grad():
                _ = layer(probe)
            return torch.device("mps")
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[device] MPS probe failed ({exc}); falling back to CPU.")
    return torch.device("cpu")


def ensure_torchaudio_backend() -> str:
    """Select a torchaudio backend so LibriSpeech FLAC files can be decoded reliably."""

    available_backends = torchaudio.list_audio_backends()
    for backend_name in ["soundfile", "ffmpeg", "sox_io"]:
        if backend_name in available_backends:
            return backend_name

    raise RuntimeError(
        "torchaudio has no available audio backend. "
        "Install soundfile or use a torchaudio build with FFmpeg support."
    )


def build_mfcc_transform() -> T.MFCC:
    """Build the MFCC extractor once so all samples use the same audio frontend."""

    return T.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": 512,
            "hop_length": 256,
            "n_mels": 64,
        },
    )


def preprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    mfcc_transform: T.MFCC,
    resamplers: Dict[int, T.Resample],
) -> torch.Tensor:
    """Convert raw audio into fixed [1, 40, 125] MFCC tensors so the CNN input is stable."""

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


def load_records_by_speaker(root: Path) -> Tuple[Dict[int, List[AudioRecord]], Dict[int, int]]:
    """Load every available utterance for the requested speakers as in-memory MFCC records."""

    root.mkdir(parents=True, exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(root),
        url=LIBRISPEECH_SPLIT,
        download=True,
    )

    raw_items: Dict[int, List[Tuple[int, int, torch.Tensor, int]]] = {speaker: [] for speaker in SPEAKER_IDS}
    for waveform, sample_rate, _, speaker_id, chapter_id, utterance_id in dataset:
        if speaker_id in raw_items:
            raw_items[speaker_id].append((chapter_id, utterance_id, waveform, sample_rate))

    speaker_to_label = {speaker_id: index for index, speaker_id in enumerate(SPEAKER_IDS)}
    mfcc_transform = build_mfcc_transform()
    resamplers: Dict[int, T.Resample] = {}
    records_by_speaker: Dict[int, List[AudioRecord]] = {}
    availability: Dict[int, int] = {}

    for speaker_id in SPEAKER_IDS:
        items = sorted(raw_items[speaker_id], key=lambda item: (item[0], item[1]))
        availability[speaker_id] = len(items)

        if len(items) < SAMPLES_PER_SPEAKER and not ALLOW_SHORT_SPEAKER_FALLBACK:
            raise ValueError(
                f"Speaker {speaker_id} has only {len(items)} utterances in {LIBRISPEECH_SPLIT}, "
                f"below the requested quota of {SAMPLES_PER_SPEAKER}."
            )

        if len(items) == 0:
            raise ValueError(f"Speaker {speaker_id} has no utterances in {LIBRISPEECH_SPLIT}.")

        speaker_records: List[AudioRecord] = []
        for chapter_id, utterance_id, waveform, sample_rate in items:
            sample_id = f"{speaker_id}-{chapter_id}-{utterance_id:04d}"
            features = preprocess_waveform(waveform, sample_rate, mfcc_transform, resamplers)
            speaker_records.append(
                AudioRecord(
                    sample_id=sample_id,
                    speaker_id=speaker_id,
                    label=speaker_to_label[speaker_id],
                    features=features,
                )
            )
        records_by_speaker[speaker_id] = speaker_records

    return records_by_speaker, availability


def partition_benchmark_records(
    all_records_by_speaker: Dict[int, List[AudioRecord]]
) -> Tuple[Dict[int, List[AudioRecord]], Dict[int, List[AudioRecord]]]:
    """Split each speaker's full pool into the benchmark subset and any extra unseen utterances."""

    benchmark_records_by_speaker: Dict[int, List[AudioRecord]] = {}
    extra_records_by_speaker: Dict[int, List[AudioRecord]] = {}
    for speaker_id, records in all_records_by_speaker.items():
        selected_count = min(SAMPLES_PER_SPEAKER, len(records))
        if selected_count < SAMPLES_PER_SPEAKER:
            print(
                f"[data] Speaker {speaker_id} has only {selected_count} utterances in {LIBRISPEECH_SPLIT}; "
                "using all available examples."
            )
        benchmark_records_by_speaker[speaker_id] = list(records[:selected_count])
        extra_records_by_speaker[speaker_id] = list(records[selected_count:])
    return benchmark_records_by_speaker, extra_records_by_speaker


def flatten_records(records_by_speaker: Dict[int, List[AudioRecord]]) -> List[AudioRecord]:
    """Flatten speaker buckets into one deterministic list for scenario construction."""

    return [record for speaker_id in SPEAKER_IDS for record in records_by_speaker[speaker_id]]


def labels_for(records: Sequence[AudioRecord]) -> List[int]:
    """Extract integer class labels so stratified splits keep speaker proportions balanced."""

    return [record.label for record in records]


def can_stratify(records: Sequence[AudioRecord]) -> bool:
    """Return whether a split can be stratified without violating class-count constraints."""

    counts = Counter(labels_for(records))
    return len(counts) > 1 and min(counts.values()) >= 2


def split_records(
    records: Sequence[AudioRecord],
    test_fraction: float,
    seed: int,
) -> Tuple[List[AudioRecord], List[AudioRecord]]:
    """Split records deterministically while stratifying whenever the label counts allow it."""

    records = list(records)
    stratify = labels_for(records) if can_stratify(records) else None
    train_records, test_records = train_test_split(
        records,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_records), list(test_records)


def split_retain_pool(
    retain_records: Sequence[AudioRecord],
    seed: int,
) -> Tuple[List[AudioRecord], List[AudioRecord], List[AudioRecord]]:
    """Split Dr into retain-train, held-out retain test, and a separate MIA holdout."""

    retain_train, remainder = split_records(
        retain_records,
        test_fraction=RETAIN_TEST_FRACTION + MIA_HOLDOUT_FRACTION,
        seed=seed,
    )
    retain_test, mia_holdout = split_records(
        remainder,
        test_fraction=MIA_HOLDOUT_FRACTION / (RETAIN_TEST_FRACTION + MIA_HOLDOUT_FRACTION),
        seed=seed + 1,
    )
    return retain_train, retain_test, mia_holdout


def build_speaker_scenario(
    records_by_speaker: Dict[int, List[AudioRecord]],
    extra_records_by_speaker: Dict[int, List[AudioRecord]],
) -> ScenarioSplit:
    """Build scenario 1 with a same-speaker MIA holdout so MIA measures membership, not speaker shift."""

    forget_records = list(records_by_speaker[FORGET_SPEAKER_ID])
    retain_records = [
        record
        for speaker_id, speaker_records in records_by_speaker.items()
        if speaker_id != FORGET_SPEAKER_ID
        for record in speaker_records
    ]

    forget_train, forget_test = split_records(
        forget_records,
        test_fraction=FORGET_TEST_FRACTION,
        seed=SEED,
    )
    retain_train, retain_test = split_records(
        retain_records,
        test_fraction=RETAIN_TEST_FRACTION,
        seed=SEED + 10,
    )
    mia_holdout = list(extra_records_by_speaker[FORGET_SPEAKER_ID])
    if not mia_holdout:
        raise ValueError(
            "Scenario 1 requires extra unseen forget-speaker utterances for a matched MIA holdout. "
            "Lower SAMPLES_PER_SPEAKER or choose a speaker with more available utterances."
        )

    return ScenarioSplit(
        name="Scenario 1",
        description=f"Df = speaker {FORGET_SPEAKER_ID}, Dr = remaining speakers",
        forget_train=forget_train,
        forget_test=forget_test,
        retain_train=retain_train,
        retain_test=retain_test,
        mia_holdout=mia_holdout,
    )


def build_sample_scenario(records_by_speaker: Dict[int, List[AudioRecord]]) -> ScenarioSplit:
    """Build scenario 2 where the forget set is a random 10% sample across speakers."""

    all_records = flatten_records(records_by_speaker)
    retain_records, forget_records = split_records(
        all_records,
        test_fraction=SAMPLE_SCENARIO_FORGET_FRACTION,
        seed=SEED + 20,
    )
    forget_train, forget_test = split_records(
        forget_records,
        test_fraction=FORGET_TEST_FRACTION,
        seed=SEED + 21,
    )
    retain_train, retain_test, mia_holdout = split_retain_pool(retain_records, seed=SEED + 22)

    return ScenarioSplit(
        name="Scenario 2",
        description="Df = random 10% of benchmark utterances across speakers",
        forget_train=forget_train,
        forget_test=forget_test,
        retain_train=retain_train,
        retain_test=retain_test,
        mia_holdout=mia_holdout,
    )


def validate_scenario(scenario: ScenarioSplit) -> None:
    """Assert split disjointness so metrics are computed on genuinely separate partitions."""

    split_to_ids = {
        "forget_train": {record.sample_id for record in scenario.forget_train},
        "forget_test": {record.sample_id for record in scenario.forget_test},
        "retain_train": {record.sample_id for record in scenario.retain_train},
        "retain_test": {record.sample_id for record in scenario.retain_test},
        "mia_holdout": {record.sample_id for record in scenario.mia_holdout},
    }

    names = list(split_to_ids)
    for index, name in enumerate(names):
        for other_name in names[index + 1 :]:
            overlap = split_to_ids[name] & split_to_ids[other_name]
            if overlap:
                raise ValueError(f"Scenario split overlap between {name} and {other_name}: {sorted(overlap)[:3]}")


def count_by_speaker(records: Sequence[AudioRecord]) -> Dict[int, int]:
    """Count samples per speaker so the printed split summary is easy to audit."""

    counts = {speaker_id: 0 for speaker_id in SPEAKER_IDS}
    for record in records:
        counts[record.speaker_id] += 1
    return counts


def describe_scenario(scenario: ScenarioSplit) -> None:
    """Print split sizes so the benchmark definition is visible in the terminal output."""

    print(f"\n[{scenario.name}] {scenario.description}")
    print(f"  forget_train : {len(scenario.forget_train):3d} {count_by_speaker(scenario.forget_train)}")
    print(f"  forget_test  : {len(scenario.forget_test):3d} {count_by_speaker(scenario.forget_test)}")
    print(f"  retain_train : {len(scenario.retain_train):3d} {count_by_speaker(scenario.retain_train)}")
    print(f"  retain_test  : {len(scenario.retain_test):3d} {count_by_speaker(scenario.retain_test)}")
    print(f"  mia_holdout  : {len(scenario.mia_holdout):3d} {count_by_speaker(scenario.mia_holdout)}")


def make_loader(records: Sequence[AudioRecord], shuffle: bool) -> DataLoader:
    """Build a DataLoader over the in-memory records for training or evaluation."""

    return DataLoader(
        RecordDataset(records),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def clone_model(source_model: nn.Module) -> TinySpeakerCNN:
    """Create a fresh model with identical weights so each method starts from the same baseline."""

    cloned_model = TinySpeakerCNN(num_classes=len(SPEAKER_IDS))
    state_dict = {name: tensor.detach().cpu().clone() for name, tensor in source_model.state_dict().items()}
    cloned_model.load_state_dict(state_dict)
    return cloned_model


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save CPU weights so checkpoints are portable across CPU and MPS executions."""

    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state_dict, path)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on one split so forget and retain utility can be compared."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else float("nan")


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    label: str,
) -> Dict[str, List[float]]:
    """Train a classifier from scratch so the original model and oracle share the same recipe."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_LR,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAIN_STEP_SIZE,
        gamma=TRAIN_GAMMA,
    )

    history = {"loss": [], "train_acc": [], "eval_acc": []}
    model.to(device)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_train_acc = correct / total
        epoch_eval_acc = evaluate_accuracy(model, eval_loader, device)
        history["loss"].append(epoch_loss)
        history["train_acc"].append(epoch_train_acc)
        history["eval_acc"].append(epoch_eval_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == TRAIN_EPOCHS:
            print(
                f"[train:{label}] epoch {epoch:02d}/{TRAIN_EPOCHS} "
                f"loss={epoch_loss:.4f} train_acc={epoch_train_acc:.3f} eval_acc={epoch_eval_acc:.3f}"
            )

    return history


def make_wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sample guaranteed incorrect labels so Random Label never reuses the true class."""

    wrong = torch.randint(0, num_classes - 1, labels.shape, device=labels.device)
    return wrong + (wrong >= labels).long()


def unlearn_gradient_ascent(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict[str, object],
) -> nn.Module:
    """Apply gradient ascent on forget data and supervised repair on retain data to remove signal."""

    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    model.to(device)
    history = {"epoch": [0], "forget_objective": [float("nan")], "retain_loss": [float("nan")]}
    analysis_loaders = config.get("analysis_loaders")
    if analysis_loaders is not None:
        baseline_snapshot = collect_unlearning_snapshot(model, analysis_loaders, device)
        append_snapshot(history, "after_forget_phase", baseline_snapshot)
        append_snapshot(history, "after_retain_phase", baseline_snapshot)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        forget_objective = 0.0
        forget_total = 0
        retain_loss = 0.0
        retain_total = 0

        for features, labels in forget_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            (-loss).backward()
            optimizer.step()
            forget_objective += loss.item() * labels.size(0)
            forget_total += labels.size(0)

        history["epoch"].append(epoch)
        history["forget_objective"].append(forget_objective / forget_total)
        history["retain_loss"].append(float("nan"))
        if analysis_loaders is not None:
            append_snapshot(
                history,
                "after_forget_phase",
                collect_unlearning_snapshot(model, analysis_loaders, device),
            )
            forget_test_after_ascent = history["after_forget_phase_forget_test_acc"][-1]

        for features, labels in retain_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            retain_loss += loss.item() * labels.size(0)
            retain_total += labels.size(0)

        history["retain_loss"][-1] = retain_loss / retain_total
        if analysis_loaders is not None:
            append_snapshot(
                history,
                "after_retain_phase",
                collect_unlearning_snapshot(model, analysis_loaders, device),
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            message = (
                f"[unlearn:GA] epoch {epoch:02d}/{config['epochs']} "
                f"forget_ce={history['forget_objective'][-1]:.4f} "
                f"retain_ce={history['retain_loss'][-1]:.4f}"
            )
            if analysis_loaders is not None:
                message += (
                    f" forget_test(after_ascent)={forget_test_after_ascent:.3f} "
                    f"forget_test(after_repair)={history['after_retain_phase_forget_test_acc'][-1]:.3f} "
                    f"retain_test(after_repair)={history['after_retain_phase_retain_test_acc'][-1]:.3f}"
                )
            print(message)

    setattr(model, "unlearning_history", history)
    return model


def unlearn_random_label(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict[str, object],
) -> nn.Module:
    """Train forget samples toward wrong classes and then refresh retain performance normally."""

    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.to(device)
    history = {"epoch": [0], "forget_wrong_loss": [float("nan")], "retain_loss": [float("nan")]}
    analysis_loaders = config.get("analysis_loaders")
    if analysis_loaders is not None:
        baseline_snapshot = collect_unlearning_snapshot(model, analysis_loaders, device)
        append_snapshot(history, "after_forget_phase", baseline_snapshot)
        append_snapshot(history, "after_retain_phase", baseline_snapshot)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        forget_wrong_loss = 0.0
        forget_total = 0
        retain_loss = 0.0
        retain_total = 0

        for features, labels in forget_loader:
            features = features.to(device)
            labels = labels.to(device)
            wrong_labels = make_wrong_labels(labels, int(config["num_classes"]))
            optimizer.zero_grad()
            loss = criterion(model(features), wrong_labels)
            loss.backward()
            optimizer.step()
            forget_wrong_loss += loss.item() * labels.size(0)
            forget_total += labels.size(0)

        history["epoch"].append(epoch)
        history["forget_wrong_loss"].append(forget_wrong_loss / forget_total)
        history["retain_loss"].append(float("nan"))
        if analysis_loaders is not None:
            append_snapshot(
                history,
                "after_forget_phase",
                collect_unlearning_snapshot(model, analysis_loaders, device),
            )
            forget_test_after_wrong_label = history["after_forget_phase_forget_test_acc"][-1]

        for features, labels in retain_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            retain_loss += loss.item() * labels.size(0)
            retain_total += labels.size(0)

        history["retain_loss"][-1] = retain_loss / retain_total
        if analysis_loaders is not None:
            append_snapshot(
                history,
                "after_retain_phase",
                collect_unlearning_snapshot(model, analysis_loaders, device),
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            message = (
                f"[unlearn:RL] epoch {epoch:02d}/{config['epochs']} "
                f"forget_wrong_ce={history['forget_wrong_loss'][-1]:.4f} retain_ce={history['retain_loss'][-1]:.4f}"
            )
            if analysis_loaders is not None:
                message += (
                    f" forget_test(after_wrong_label)={forget_test_after_wrong_label:.3f} "
                    f"forget_test(after_repair)={history['after_retain_phase_forget_test_acc'][-1]:.3f} "
                    f"retain_test(after_repair)={history['after_retain_phase_retain_test_acc'][-1]:.3f}"
                )
            print(message)

    setattr(model, "unlearning_history", history)
    return model


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions and labels so confusion matrices and per-speaker accuracy share one pass."""

    model.eval()
    all_predictions: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    return predictions, labels


def collect_per_sample_losses(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Collect per-sample cross-entropy losses so the MIA uses a proper threshold attack."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses: List[torch.Tensor] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            losses = criterion(model(features), labels)
            all_losses.append(losses.cpu())

    return torch.cat(all_losses).numpy()


def mean_cross_entropy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return mean cross-entropy on a split so unlearning diagnostics can track confidence changes."""

    losses = collect_per_sample_losses(model, loader, device)
    return float(losses.mean())


def collect_unlearning_snapshot(
    model: nn.Module,
    analysis_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """Collect split-wise accuracy and loss so unlearning dynamics can be inspected over time."""

    return {
        "forget_train_acc": evaluate_accuracy(model, analysis_loaders["forget_train"], device),
        "forget_test_acc": evaluate_accuracy(model, analysis_loaders["forget_test"], device),
        "retain_test_acc": evaluate_accuracy(model, analysis_loaders["retain_test"], device),
        "full_test_acc": evaluate_accuracy(model, analysis_loaders["full_test"], device),
        "forget_train_ce": mean_cross_entropy(model, analysis_loaders["forget_train"], device),
        "forget_test_ce": mean_cross_entropy(model, analysis_loaders["forget_test"], device),
    }


def append_snapshot(history: Dict[str, List[float]], prefix: str, snapshot: Dict[str, float]) -> None:
    """Append one diagnostic snapshot under a named phase without hardcoding every metric key."""

    for metric_name, metric_value in snapshot.items():
        history.setdefault(f"{prefix}_{metric_name}", []).append(metric_value)


def collect_mia_scores(
    model: nn.Module,
    forget_member_loader: DataLoader,
    mia_holdout_loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Return MIA member and non-member scores so AUC and histograms use the same losses."""

    member_losses = collect_per_sample_losses(model, forget_member_loader, device)
    non_member_losses = collect_per_sample_losses(model, mia_holdout_loader, device)
    return {
        "member_scores": -member_losses,
        "non_member_scores": -non_member_losses,
    }


def evaluate_all(
    model: nn.Module,
    forget_test_loader: DataLoader,
    retain_test_loader: DataLoader,
    full_test_loader: DataLoader,
    forget_member_loader: DataLoader,
    mia_holdout_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute the four benchmark metrics so every method is compared on one consistent definition."""

    forget_accuracy = evaluate_accuracy(model, forget_test_loader, device)
    retain_accuracy = evaluate_accuracy(model, retain_test_loader, device)
    test_utility = evaluate_accuracy(model, full_test_loader, device)

    member_losses = collect_per_sample_losses(model, forget_member_loader, device)
    non_member_losses = collect_per_sample_losses(model, mia_holdout_loader, device)
    mia_labels = np.concatenate(
        [np.ones(len(member_losses), dtype=np.int64), np.zeros(len(non_member_losses), dtype=np.int64)]
    )
    mia_scores = np.concatenate([-member_losses, -non_member_losses])
    mia_auc = roc_auc_score(mia_labels, mia_scores)

    return {
        "Forget Acc ↓": forget_accuracy,
        "Retain Acc ↑": retain_accuracy,
        "MIA AUC ↓": mia_auc,
        "Test Utility ↑": test_utility,
    }


def compute_per_speaker_accuracy(
    model: nn.Module,
    records: Sequence[AudioRecord],
    device: torch.device,
) -> Dict[int, float]:
    """Measure accuracy per speaker on the held-out test set so forgetting is visible by identity."""

    loader = make_loader(records, shuffle=False)
    predictions, labels = collect_predictions(model, loader, device)
    speaker_ids = np.array([record.speaker_id for record in records])
    result: Dict[int, float] = {}

    for speaker_id in SPEAKER_IDS:
        mask = speaker_ids == speaker_id
        if mask.any():
            result[speaker_id] = float((predictions[mask] == labels[mask]).mean())
        else:
            result[speaker_id] = float("nan")

    return result


def format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Format the numeric results table for terminal readability without changing saved CSV values."""

    formatted = results_df.copy()
    for column in ["Forget Acc ↓", "Retain Acc ↑", "Test Utility ↑"]:
        formatted[column] = (formatted[column] * 100.0).map(lambda value: f"{value:6.2f}%")
    formatted["MIA AUC ↓"] = formatted["MIA AUC ↓"].map(lambda value: f"{value:.4f}")
    return formatted


def plot_results(
    original_history: Dict[str, List[float]],
    retrain_history: Dict[str, List[float]],
    speaker_accuracies: Dict[str, Dict[int, float]],
    confusion_payloads: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mia_payloads: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Save the requested 4-panel summary figure so the demo has one compact artifact."""

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    fig.suptitle("Speech Unlearning Benchmark", fontsize=16, fontweight="bold")

    ax_loss = fig.add_subplot(grid[0, 0])
    ax_loss.plot(original_history["loss"], label="Original Train Loss", color="#1f77b4", linewidth=2)
    ax_loss.plot(retrain_history["loss"], label="Retrain Oracle Loss", color="#2ca02c", linewidth=2)
    ax_loss.set_title("Training Loss Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_bar = fig.add_subplot(grid[0, 1])
    methods_for_bar = ["No Unlearning", "Gradient Ascent", "Random Label"]
    colors = {
        "No Unlearning": "#1f77b4",
        "Gradient Ascent": "#d62728",
        "Random Label": "#ff7f0e",
        "Retrain Oracle": "#2ca02c",
    }
    x_positions = np.arange(len(SPEAKER_IDS))
    width = 0.24
    offsets = [-width, 0.0, width]
    for offset, method_name in zip(offsets, methods_for_bar):
        values = [speaker_accuracies[method_name][speaker_id] * 100.0 for speaker_id in SPEAKER_IDS]
        ax_bar.bar(
            x_positions + offset,
            values,
            width=width,
            label=method_name,
            color=colors[method_name],
            alpha=0.85,
        )
    forget_index = SPEAKER_IDS.index(FORGET_SPEAKER_ID)
    ax_bar.axvspan(forget_index - 0.5, forget_index + 0.5, color="#fddede", alpha=0.35)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([str(speaker_id) for speaker_id in SPEAKER_IDS])
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title("Per-Speaker Test Accuracy")
    ax_bar.set_xlabel("Speaker ID")
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend()

    confusion_outer = fig.add_subplot(grid[1, 0])
    confusion_outer.axis("off")
    confusion_outer.set_title("Confusion Matrices", y=1.02)
    confusion_grid = grid[1, 0].subgridspec(1, 3, wspace=0.35)
    for index, method_name in enumerate(["No Unlearning", "Gradient Ascent", "Random Label"]):
        ax_cm = fig.add_subplot(confusion_grid[0, index])
        predictions, labels = confusion_payloads[method_name]
        cm = confusion_matrix(labels, predictions, labels=np.arange(len(SPEAKER_IDS)))
        disp = ConfusionMatrixDisplay(cm, display_labels=[str(speaker_id) for speaker_id in SPEAKER_IDS])
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues", values_format="d")
        ax_cm.set_title(method_name, fontsize=10)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")

    ax_mia = fig.add_subplot(grid[1, 1])
    all_scores = []
    for payload in mia_payloads.values():
        all_scores.extend(payload["member_scores"].tolist())
        all_scores.extend(payload["non_member_scores"].tolist())
    bins = np.linspace(min(all_scores), max(all_scores), 25)

    for method_name in METHOD_NAMES:
        payload = mia_payloads[method_name]
        ax_mia.hist(
            payload["member_scores"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=colors[method_name],
            label=f"{method_name} member",
        )
        ax_mia.hist(
            payload["non_member_scores"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            linestyle="--",
            color=colors[method_name],
            label=f"{method_name} non-member",
        )

    ax_mia.set_title("MIA Score Distributions (-CE Loss)")
    ax_mia.set_xlabel("Membership Score")
    ax_mia.set_ylabel("Density")
    ax_mia.grid(True, alpha=0.3)
    ax_mia.legend(fontsize=8, ncol=2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_history_table(history: Dict[str, List[float]], output_path: Path) -> None:
    """Persist unlearning history so diagnostics can be inspected numerically outside the plot."""

    pd.DataFrame(history).to_csv(output_path, index=False)


def plot_stepwise_diagnostics(
    history: Dict[str, List[float]],
    output_path: Path,
    method_title: str,
    phase_name: str,
    objective_key: str,
    objective_label: str,
) -> None:
    """Plot per-epoch forgetting/repair diagnostics for any unlearning method using the shared history schema."""

    epochs = np.array(history["epoch"], dtype=float)
    objective_epochs = epochs[1:]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{method_title} Step-by-Step Diagnostics", fontsize=15, fontweight="bold")

    axes[0, 0].plot(objective_epochs, history[objective_key][1:], color="#d62728", linewidth=2, label=objective_label)
    axes[0, 0].plot(objective_epochs, history["retain_loss"][1:], color="#1f77b4", linewidth=2, label="Retain Repair CE")
    axes[0, 0].set_title("Optimization Objectives")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs,
        history["after_forget_phase_forget_train_acc"],
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Forget Train Acc after {phase_name}",
    )
    axes[0, 1].plot(
        epochs,
        history["after_retain_phase_forget_train_acc"],
        color="#d62728",
        linewidth=2,
        label="Forget Train Acc after Repair",
    )
    axes[0, 1].plot(
        epochs,
        history["after_forget_phase_forget_test_acc"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=f"Forget Test Acc after {phase_name}",
    )
    axes[0, 1].plot(
        epochs,
        history["after_retain_phase_forget_test_acc"],
        color="#ff7f0e",
        linewidth=2,
        label="Forget Test Acc after Repair",
    )
    axes[0, 1].set_title("Forget Accuracy Trajectory")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        epochs,
        history["after_forget_phase_forget_train_ce"],
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Forget Train CE after {phase_name}",
    )
    axes[1, 0].plot(
        epochs,
        history["after_retain_phase_forget_train_ce"],
        color="#d62728",
        linewidth=2,
        label="Forget Train CE after Repair",
    )
    axes[1, 0].plot(
        epochs,
        history["after_forget_phase_forget_test_ce"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=f"Forget Test CE after {phase_name}",
    )
    axes[1, 0].plot(
        epochs,
        history["after_retain_phase_forget_test_ce"],
        color="#ff7f0e",
        linewidth=2,
        label="Forget Test CE after Repair",
    )
    axes[1, 0].set_title("Forget Cross-Entropy Trajectory")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cross-Entropy")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        epochs,
        history["after_forget_phase_retain_test_acc"],
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=f"Retain Test Acc after {phase_name}",
    )
    axes[1, 1].plot(
        epochs,
        history["after_retain_phase_retain_test_acc"],
        color="#1f77b4",
        linewidth=2,
        label="Retain Test Acc after Repair",
    )
    axes[1, 1].plot(
        epochs,
        history["after_forget_phase_full_test_acc"],
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"Full Test Acc after {phase_name}",
    )
    axes[1, 1].plot(
        epochs,
        history["after_retain_phase_full_test_acc"],
        color="#2ca02c",
        linewidth=2,
        label="Full Test Acc after Repair",
    )
    axes[1, 1].set_title("Retain / Overall Utility")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ga_stepwise(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plot GA before/after-repair trajectories so failed forgetting can be diagnosed directly."""

    plot_stepwise_diagnostics(
        history=history,
        output_path=output_path,
        method_title="Gradient Ascent",
        phase_name="Ascent",
        objective_key="forget_objective",
        objective_label="Forget CE",
    )


def plot_rl_stepwise(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plot RL before/after-repair trajectories so wrong-label forgetting can be diagnosed directly."""

    plot_stepwise_diagnostics(
        history=history,
        output_path=output_path,
        method_title="Random Label",
        phase_name="Wrong-Label",
        objective_key="forget_wrong_loss",
        objective_label="Wrong-Label CE",
    )


def main() -> None:
    """Run the full benchmark end to end so the demo is reproducible from one script."""

    set_seed(SEED)
    BENCHMARK_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    audio_backend = ensure_torchaudio_backend()
    print(f"[setup] device={device}")
    print(f"[setup] torchaudio_backend={audio_backend}")
    print(f"[setup] LibriSpeech split={LIBRISPEECH_SPLIT}")
    print(f"[setup] target speakers={SPEAKER_IDS}")

    all_records_by_speaker, availability = load_records_by_speaker(DATA_ROOT)
    records_by_speaker, extra_records_by_speaker = partition_benchmark_records(all_records_by_speaker)
    actual_counts = {speaker_id: len(records_by_speaker[speaker_id]) for speaker_id in SPEAKER_IDS}
    print(f"[data] available utterances per speaker={availability}")
    print(f"[data] benchmark utterances per speaker={actual_counts}")
    print(f"[data] total benchmark utterances={sum(actual_counts.values())}")

    scenario_1 = build_speaker_scenario(records_by_speaker, extra_records_by_speaker)
    scenario_2 = build_sample_scenario(records_by_speaker)
    validate_scenario(scenario_1)
    validate_scenario(scenario_2)
    describe_scenario(scenario_1)
    print(f"\n[{scenario_2.name}] implemented but not run by default: {scenario_2.description}")

    scenario = scenario_1 if SCENARIO_TO_RUN == 1 else scenario_2
    if SCENARIO_TO_RUN not in {1, 2}:
        raise ValueError(f"Unsupported SCENARIO_TO_RUN={SCENARIO_TO_RUN}")

    print(f"\n[run] executing {scenario.name}: {scenario.description}")

    original_train_loader = make_loader(scenario.original_train, shuffle=True)
    retain_train_loader = make_loader(scenario.retain_train, shuffle=True)
    forget_train_loader = make_loader(scenario.forget_train, shuffle=True)

    forget_train_eval_loader = make_loader(scenario.forget_train, shuffle=False)
    forget_test_loader = make_loader(scenario.forget_test, shuffle=False)
    retain_test_loader = make_loader(scenario.retain_test, shuffle=False)
    full_test_loader = make_loader(scenario.full_test, shuffle=False)
    mia_holdout_loader = make_loader(scenario.mia_holdout, shuffle=False)
    analysis_loaders = {
        "forget_train": forget_train_eval_loader,
        "forget_test": forget_test_loader,
        "retain_test": retain_test_loader,
        "full_test": full_test_loader,
    }

    original_model = TinySpeakerCNN()
    original_history = train_supervised(
        original_model,
        original_train_loader,
        full_test_loader,
        device,
        label="original",
    )
    save_checkpoint(original_model, ORIGINAL_CHECKPOINT)

    retrain_model = TinySpeakerCNN()
    retrain_history = train_supervised(
        retrain_model,
        retain_train_loader,
        full_test_loader,
        device,
        label="retrain",
    )
    save_checkpoint(retrain_model, RETRAIN_CHECKPOINT)

    ga_model = clone_model(original_model)
    ga_model = unlearn_gradient_ascent(
        ga_model,
        forget_train_loader,
        retain_train_loader,
        config={
            "epochs": GA_EPOCHS,
            "lr": GA_LR,
            "device": device,
            "analysis_loaders": analysis_loaders,
        },
    )

    rl_model = clone_model(original_model)
    rl_model = unlearn_random_label(
        rl_model,
        forget_train_loader,
        retain_train_loader,
        config={
            "epochs": RL_EPOCHS,
            "lr": RL_LR,
            "device": device,
            "num_classes": len(SPEAKER_IDS),
            "analysis_loaders": analysis_loaders,
        },
    )

    models = {
        "No Unlearning": original_model,
        "Retrain Oracle": retrain_model,
        "Gradient Ascent": ga_model,
        "Random Label": rl_model,
    }

    results_rows = []
    speaker_accuracies: Dict[str, Dict[int, float]] = {}
    confusion_payloads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    mia_payloads: Dict[str, Dict[str, np.ndarray]] = {}

    for method_name in METHOD_NAMES:
        model = models[method_name]
        metrics = evaluate_all(
            model,
            forget_test_loader,
            retain_test_loader,
            full_test_loader,
            forget_train_eval_loader,
            mia_holdout_loader,
            device,
        )
        results_rows.append(metrics)
        speaker_accuracies[method_name] = compute_per_speaker_accuracy(model, scenario.full_test, device)
        if method_name != "Retrain Oracle":
            confusion_payloads[method_name] = collect_predictions(model, full_test_loader, device)
        mia_payloads[method_name] = collect_mia_scores(
            model,
            forget_train_eval_loader,
            mia_holdout_loader,
            device,
        )

    results_df = pd.DataFrame(results_rows, index=METHOD_NAMES)
    results_df.to_csv(RESULTS_CSV)
    print("\nResults Summary")
    print(format_results_table(results_df).to_string())
    print("[note] For MIA AUC, values closer to 0.5 mean the loss-based attack is closer to random guessing.")
    print("[note] Values below 0.5 mean the chosen score direction separates the sets in the opposite direction.")

    plot_results(
        original_history=original_history,
        retrain_history=retrain_history,
        speaker_accuracies=speaker_accuracies,
        confusion_payloads=confusion_payloads,
        mia_payloads=mia_payloads,
        output_path=RESULTS_FIGURE,
    )
    save_history_table(getattr(ga_model, "unlearning_history"), GA_STEPWISE_CSV)
    plot_ga_stepwise(getattr(ga_model, "unlearning_history"), GA_STEPWISE_FIGURE)
    save_history_table(getattr(rl_model, "unlearning_history"), RL_STEPWISE_CSV)
    plot_rl_stepwise(getattr(rl_model, "unlearning_history"), RL_STEPWISE_FIGURE)

    print(f"\n[artifacts] saved {ORIGINAL_CHECKPOINT}")
    print(f"[artifacts] saved {RETRAIN_CHECKPOINT}")
    print(f"[artifacts] saved {RESULTS_CSV}")
    print(f"[artifacts] saved {RESULTS_FIGURE}")
    print(f"[artifacts] saved {GA_STEPWISE_CSV}")
    print(f"[artifacts] saved {GA_STEPWISE_FIGURE}")
    print(f"[artifacts] saved {RL_STEPWISE_CSV}")
    print(f"[artifacts] saved {RL_STEPWISE_FIGURE}")


if __name__ == "__main__":
    main()
