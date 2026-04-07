"""Multimodal celebrity speaker unlearning benchmark.

Scenario A: unlearn the audio branch only while freezing the face branch.
Scenario B: unlearn both modalities jointly.

Expected assets
---------------
- Audio clips under data/celebrity/<name>/*.wav
- Companion videos under data/celebrity_video/<name>/<sample_id>.<ext>
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_pipeline.celebrity_loader import (
    ALL_CELEBRITIES,
    FORGET_CELEBRITY,
    build_celebrity_scenario,
    load_celebrity_records,
)
from data_pipeline.face_loader import load_multimodal_records
from models.multimodal import MultimodalSpeakerCNN

REPO_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = REPO_ROOT / "data" / "celebrity"
VIDEO_DIR = REPO_ROOT / "data" / "celebrity_video"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "multimodal"

NUM_CLASSES = len(ALL_CELEBRITIES)
BATCH_SIZE = 16
NUM_WORKERS = 0

TRAIN_EPOCHS = 60
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4

UNLEARN_EPOCHS = 20
UNLEARN_LR = 1e-4

METHODS = [
    "gradient_ascent",
    "random_label",
    "fine_tune",
    "scrub",
    "ssd",
]


class MultimodalRecordDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return record.audio_features, record.face_features, record.label


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        try:
            probe = torch.randn(1, 1, 40, 125, device="mps")
            layer = nn.Sequential(
                nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            ).to("mps")
            with torch.no_grad():
                layer(probe)
            return torch.device("mps")
        except Exception as exc:
            print(f"[device] MPS probe failed ({exc}); falling back to CPU.")
    return torch.device("cpu")


def make_loader(records, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        MultimodalRecordDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def clone_model(source: nn.Module) -> MultimodalSpeakerCNN:
    model = MultimodalSpeakerCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(copy.deepcopy(source.state_dict()))
    return model


def _make_wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    wrong = torch.randint(0, num_classes - 1, labels.shape, device=labels.device)
    return wrong + (wrong >= labels).long()


def _set_face_branch_frozen(model: MultimodalSpeakerCNN, frozen: bool) -> None:
    for parameter in model.face_encoder.parameters():
        parameter.requires_grad_(not frozen)


def _optimizer_for_trainable_params(model: nn.Module, lr: float, weight_decay: float = 0.0):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _prepare_train_mode(model: MultimodalSpeakerCNN, freeze_face_encoder: bool) -> None:
    model.train()
    if freeze_face_encoder:
        model.face_encoder.eval()


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for audio_features, face_features, labels in loader:
            audio_features = audio_features.to(device)
            face_features = face_features.to(device)
            labels = labels.to(device)
            logits = model(audio_features, face_features)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else float("nan")


def evaluate_all(
    model: nn.Module,
    forget_test_loader: DataLoader,
    retain_test_loader: DataLoader,
    full_test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    return {
        "Forget Acc ↓": evaluate_accuracy(model, forget_test_loader, device),
        "Retain Acc ↑": evaluate_accuracy(model, retain_test_loader, device),
        "Test Utility ↑": evaluate_accuracy(model, full_test_loader, device),
    }


def train_supervised(
    model: MultimodalSpeakerCNN,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    label: str,
    epochs: int,
) -> MultimodalSpeakerCNN:
    criterion = nn.CrossEntropyLoss()
    optimizer = _optimizer_for_trainable_params(
        model,
        lr=TRAIN_LR,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for audio_features, face_features, labels in train_loader:
            audio_features = audio_features.to(device)
            face_features = face_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(audio_features, face_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            eval_acc = evaluate_accuracy(model, eval_loader, device)
            print(
                f"[train:{label}] epoch {epoch:02d}/{epochs} "
                f"loss={running_loss/total:.4f} "
                f"train_acc={correct/total:.3f} "
                f"eval_acc={eval_acc:.3f}"
            )

    return model


def _compute_fisher_diagonal(
    model: MultimodalSpeakerCNN,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    freeze_face_encoder: bool,
) -> Dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    fisher = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    total = 0

    for audio_features, face_features, labels in loader:
        if total >= n_samples:
            break

        audio_features = audio_features.to(device)
        face_features = face_features.to(device)
        labels = labels.to(device)
        model.zero_grad()
        _prepare_train_mode(model, freeze_face_encoder)
        loss = criterion(model(audio_features, face_features), labels)
        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2 * labels.size(0)

        total += labels.size(0)

    for name in fisher:
        fisher[name] /= max(total, 1)

    return fisher


def _apply_method(
    model: MultimodalSpeakerCNN,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    method: str,
    config: Dict,
    *,
    freeze_face_encoder: bool,
) -> MultimodalSpeakerCNN:
    if method == "sisa":
        raise ValueError("SISA is not supported in the multimodal benchmark yet.")

    model.to(device)
    _set_face_branch_frozen(model, freeze_face_encoder)

    if method == "ssd":
        alpha = float(config.get("ssd_alpha", 1.0))
        eps = float(config.get("ssd_eps", 1e-8))
        n_samples = int(config.get("ssd_n", 200))
        print(f"[{method}] computing Fisher diagonals...")
        fisher_forget = _compute_fisher_diagonal(
            model,
            forget_loader,
            device,
            n_samples=n_samples,
            freeze_face_encoder=freeze_face_encoder,
        )
        fisher_retain = _compute_fisher_diagonal(
            model,
            retain_loader,
            device,
            n_samples=n_samples,
            freeze_face_encoder=freeze_face_encoder,
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                ratio = fisher_forget[name] / (fisher_retain[name] + eps)
                parameter.data -= alpha * ratio * parameter.data
        return model

    criterion = nn.CrossEntropyLoss()
    optimizer = _optimizer_for_trainable_params(model, lr=config["lr"])
    teacher = None
    if method == "scrub":
        teacher = clone_model(config["teacher_model"]).to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    for epoch in range(1, int(config["epochs"]) + 1):
        _prepare_train_mode(model, freeze_face_encoder)
        forget_value = 0.0
        forget_n = 0
        retain_value = 0.0
        retain_n = 0

        if method != "fine_tune":
            for audio_features, face_features, labels in forget_loader:
                audio_features = audio_features.to(device)
                face_features = face_features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if method == "gradient_ascent":
                    loss = criterion(model(audio_features, face_features), labels)
                    (-loss).backward()
                    step_value = loss
                elif method == "random_label":
                    wrong = _make_wrong_labels(labels, int(config["num_classes"]))
                    loss = criterion(model(audio_features, face_features), wrong)
                    loss.backward()
                    step_value = loss
                elif method == "scrub":
                    temperature = float(config.get("temperature", 4.0))
                    with torch.no_grad():
                        teacher_logits = teacher(audio_features, face_features) / temperature
                    student_logits = model(audio_features, face_features) / temperature
                    loss = F.kl_div(
                        F.log_softmax(student_logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1),
                        reduction="batchmean",
                    )
                    (-loss).backward()
                    step_value = loss
                else:
                    raise ValueError(f"Unsupported method: {method}")

                optimizer.step()
                forget_value += step_value.item() * labels.size(0)
                forget_n += labels.size(0)

        for audio_features, face_features, labels in retain_loader:
            audio_features = audio_features.to(device)
            face_features = face_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if method == "scrub":
                temperature = float(config.get("temperature", 4.0))
                with torch.no_grad():
                    teacher_logits = teacher(audio_features, face_features) / temperature
                student_logits = model(audio_features, face_features) / temperature
                kl = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                )
                ce = criterion(model(audio_features, face_features), labels)
                loss = float(config.get("alpha", 1.0)) * kl + ce
                retain_step_value = kl
            else:
                loss = criterion(model(audio_features, face_features), labels)
                retain_step_value = loss

            loss.backward()
            optimizer.step()
            retain_value += retain_step_value.item() * labels.size(0)
            retain_n += labels.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            forget_metric = forget_value / forget_n if forget_n else float("nan")
            retain_metric = retain_value / retain_n if retain_n else float("nan")
            print(
                f"[{method}] epoch {epoch:02d}/{config['epochs']} "
                f"forget={forget_metric:.4f} "
                f"retain={retain_metric:.4f}"
            )

    return model


def unlearn_audio_only(
    multimodal_model: MultimodalSpeakerCNN,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    method: str,
    config: Dict,
) -> MultimodalSpeakerCNN:
    return _apply_method(
        multimodal_model,
        forget_loader,
        retain_loader,
        config["device"],
        method,
        config,
        freeze_face_encoder=True,
    )


def unlearn_jointly(
    multimodal_model: MultimodalSpeakerCNN,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    method: str,
    config: Dict,
) -> MultimodalSpeakerCNN:
    return _apply_method(
        multimodal_model,
        forget_loader,
        retain_loader,
        config["device"],
        method,
        config,
        freeze_face_encoder=False,
    )


def _save_results_plot(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        metrics = [column for column in df.columns if column not in {"Scenario", "Method"}]
        n_cols = min(3, len(metrics))
        n_rows = math.ceil(len(metrics) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols)

        for ax, metric in zip(axes.flat, metrics):
            pivot = df.pivot(index="Method", columns="Scenario", values=metric)
            pivot.plot(kind="bar", ax=ax, rot=25, width=0.8)
            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
            ax.legend(loc="best", fontsize=8)

        for ax in axes.flat[len(metrics):]:
            ax.axis("off")

        fig.suptitle("Multimodal Unlearning: Audio-Only vs Joint", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[benchmark] plot saved to {out_path}")
    except Exception as exc:
        print(f"[benchmark] WARN: could not save results plot: {exc}")


def run_benchmark(args: argparse.Namespace) -> None:
    set_seed(42)
    device = select_device()
    print(f"[benchmark] device={device}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    samples = 10 if args.quick else 50
    train_epochs = 5 if args.quick else TRAIN_EPOCHS
    unlearn_epochs = 3 if args.quick else UNLEARN_EPOCHS

    audio_records = load_celebrity_records(
        args.audio_dir,
        speakers=ALL_CELEBRITIES,
        samples_per_speaker=samples,
    )
    multimodal_records = load_multimodal_records(
        audio_records,
        args.video_dir,
        num_frames=5,
        image_size=96,
    )

    missing = [name for name, records in multimodal_records.items() if not records]
    if missing:
        print(
            f"\n[benchmark] ERROR: no multimodal clips found for: {missing}\n"
            f"  Expected audio under {args.audio_dir}/<name>/*.wav\n"
            f"  Expected videos under {args.video_dir}/<name>/<sample_id>.<ext>\n"
        )
        sys.exit(1)

    scenario = build_celebrity_scenario(multimodal_records, forget_speaker=FORGET_CELEBRITY)
    print(
        f"[benchmark] scenario: "
        f"forget_train={len(scenario.forget_train)} "
        f"forget_test={len(scenario.forget_test)} "
        f"retain_train={len(scenario.retain_train)} "
        f"retain_test={len(scenario.retain_test)}"
    )

    train_loader = make_loader(scenario.original_train, shuffle=True)
    forget_train_loader = make_loader(scenario.forget_train, shuffle=True)
    forget_test_loader = make_loader(scenario.forget_test, shuffle=False)
    retain_train_loader = make_loader(scenario.retain_train, shuffle=True)
    retain_test_loader = make_loader(scenario.retain_test, shuffle=False)
    full_test_loader = make_loader(scenario.full_test, shuffle=False)

    print("\n[benchmark] === Training multimodal baseline ===")
    original_model = MultimodalSpeakerCNN(num_classes=NUM_CLASSES)
    train_supervised(
        original_model,
        train_loader,
        full_test_loader,
        device,
        label="multimodal",
        epochs=train_epochs,
    )

    rows = [{
        "Scenario": "baseline",
        "Method": "No Unlearning",
        **evaluate_all(original_model, forget_test_loader, retain_test_loader, full_test_loader, device),
    }]

    base_config = {
        "device": device,
        "lr": UNLEARN_LR,
        "epochs": unlearn_epochs,
        "num_classes": NUM_CLASSES,
    }
    scrub_config = {
        **base_config,
        "teacher_model": original_model,
        "alpha": 1.0,
        "temperature": 4.0,
    }
    ssd_config = {
        **base_config,
        "ssd_alpha": 1.0,
        "ssd_eps": 1e-8,
        "ssd_n": 200,
    }

    for method in METHODS:
        config = scrub_config if method == "scrub" else ssd_config if method == "ssd" else base_config

        print(f"\n[benchmark] === Scenario A: {method} (audio-only) ===")
        audio_only_model = clone_model(original_model)
        audio_only_model = unlearn_audio_only(
            audio_only_model,
            forget_train_loader,
            retain_train_loader,
            method,
            config,
        )
        rows.append({
            "Scenario": "audio_only",
            "Method": method.replace("_", " ").title(),
            **evaluate_all(audio_only_model, forget_test_loader, retain_test_loader, full_test_loader, device),
        })

        print(f"\n[benchmark] === Scenario B: {method} (joint) ===")
        joint_model = clone_model(original_model)
        joint_model = unlearn_jointly(
            joint_model,
            forget_train_loader,
            retain_train_loader,
            method,
            config,
        )
        rows.append({
            "Scenario": "joint",
            "Method": method.replace("_", " ").title(),
            **evaluate_all(joint_model, forget_test_loader, retain_test_loader, full_test_loader, device),
        })

    df = pd.DataFrame(rows)
    print("\n=== Results ===")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    csv_path = ARTIFACT_DIR / "results_summary.csv"
    plot_path = ARTIFACT_DIR / "results.png"
    df.to_csv(csv_path, index=False)
    print(f"\n[benchmark] results saved to {csv_path}")
    _save_results_plot(df, plot_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal celebrity unlearning benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 5 epochs, 10 clips per speaker",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help="Directory containing celebrity WAV clips",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help="Directory containing companion celebrity videos",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(_parse_args())
