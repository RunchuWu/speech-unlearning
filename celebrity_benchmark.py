"""Celebrity speaker unlearning benchmark.

Trains a TinySpeakerCNN on five celebrity speakers (Trump as forget target),
applies six unlearning methods, and reports a comparative results table.

Usage
-----
    python celebrity_benchmark.py            # full run
    python celebrity_benchmark.py --quick    # smoke test (5 epochs, 10 clips)
    python celebrity_benchmark.py --download # download clips first, then run

Outputs saved to artifacts/celebrity/
    results_summary.csv
    results.png
    before_after_tsne.png  (if evaluation/visualization.py is present)
    parameter_change_<method>.png
    models/*.pt
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
from torch.utils.data import DataLoader

from data_pipeline.celebrity_loader import (
    CELEBRITY_MANIFEST,
    FORGET_CELEBRITY,
    RETAIN_CELEBRITIES,
    ALL_CELEBRITIES,
    RecordDataset,
    build_celebrity_scenario,
    download_celebrity_clips,
    load_celebrity_records,
)
from models.audio_cnn import TinySpeakerCNN
from unlearning.methods import (
    REGISTRY,
    train_sisa_shards,
)
from evaluation import (
    compute_eer,
    compute_weight_distance,
    label_only_mia,
    loss_threshold_mia,
    plot_before_after_tsne_grid,
    plot_parameter_change_heatmap,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "celebrity"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "celebrity"
MODEL_DIR = ARTIFACT_DIR / "models"

NUM_CLASSES = len(ALL_CELEBRITIES)  # 5
BATCH_SIZE = 16
NUM_WORKERS = 0

TRAIN_EPOCHS = 60
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_STEP_SIZE = 20
TRAIN_GAMMA = 0.5

UNLEARN_EPOCHS = 20
UNLEARN_LR = 1e-4

SISA_SHARDS = 4


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
        RecordDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


class SISAEnsemble(nn.Module):
    """Average logits and embeddings across SISA shard models."""

    def __init__(self, members: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = [member(inputs) for member in self.members]
        return torch.stack(logits, dim=0).mean(dim=0)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for member in self.members:
            if hasattr(member, "embed"):
                embeddings.append(member.embed(inputs))
            else:
                embeddings.append(member(inputs))
        return torch.stack(embeddings, dim=0).mean(dim=0)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            correct += (model(features).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else float("nan")


def _weight_distance_scalar(model: nn.Module, oracle_model: nn.Module) -> float:
    """Reduce per-layer distances to one comparable scalar for the summary table."""
    if isinstance(model, SISAEnsemble):
        shard_distances = []
        for member in model.members:
            distances = compute_weight_distance(member, oracle_model)
            if distances:
                shard_distances.append(sum(distances.values()))
        return float(np.mean(shard_distances)) if shard_distances else float("nan")

    distances = compute_weight_distance(model, oracle_model)
    return float(sum(distances.values())) if distances else float("nan")


def evaluate_all(
    model: nn.Module,
    forget_test_loader: DataLoader,
    retain_test_loader: DataLoader,
    full_test_loader: DataLoader,
    forget_member_records: List,
    mia_holdout_records: List,
    evaluation_records: List,
    oracle_model: nn.Module,
    device: torch.device,
    *,
    eer_pairs: int = 1000,
    label_only_n_aug: int = 10,
) -> Dict[str, float]:
    forget_acc = evaluate_accuracy(model, forget_test_loader, device)
    retain_acc = evaluate_accuracy(model, retain_test_loader, device)
    test_utility = evaluate_accuracy(model, full_test_loader, device)

    loss_mia = loss_threshold_mia(model, forget_member_records, mia_holdout_records, device)
    label_mia = label_only_mia(
        model,
        forget_member_records,
        mia_holdout_records,
        device,
        n_aug=label_only_n_aug,
    )
    eer = compute_eer(model, evaluation_records, device, num_pairs=eer_pairs)
    weight_distance = _weight_distance_scalar(model, oracle_model)

    return {
        "Forget Acc ↓": forget_acc,
        "Retain Acc ↑": retain_acc,
        "MIA AUC ↓": loss_mia["mia_auc"],
        "Label-Only MIA AUC ↓": label_mia["mia_auc"],
        "Test Utility ↑": test_utility,
        "EER ↓": eer,
        "Weight L2 vs Oracle ↓": weight_distance,
    }


def clone_model(source: nn.Module, num_classes: int = NUM_CLASSES) -> TinySpeakerCNN:
    m = TinySpeakerCNN(num_classes=num_classes)
    m.load_state_dict(copy.deepcopy(source.state_dict()))
    return m


def _checkpoint_payload(model: nn.Module) -> Dict:
    return {
        "model_type": "audio_cnn",
        "num_classes": NUM_CLASSES,
        "label_names": ALL_CELEBRITIES,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    label: str,
    epochs: int = TRAIN_EPOCHS,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=TRAIN_STEP_SIZE, gamma=TRAIN_GAMMA)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            eval_acc = evaluate_accuracy(model, eval_loader, device)
            print(
                f"[train:{label}] epoch {epoch:02d}/{epochs} "
                f"loss={running_loss/total:.4f} "
                f"train_acc={correct/total:.3f} "
                f"eval_acc={eval_acc:.3f}"
            )

    return model


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    set_seed(42)
    device = select_device()
    print(f"[benchmark] device={device}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Download (optional) ----
    if args.download:
        print("[benchmark] downloading celebrity clips...")
        for name, entries in CELEBRITY_MANIFEST.items():
            if entries:
                download_celebrity_clips(name, entries, DATA_DIR / name)
            else:
                print(f"[download] WARN: no manifest entries for {name} — skipping")

    # ---- Load data ----
    samples = 10 if args.quick else 50
    records = load_celebrity_records(DATA_DIR, speakers=ALL_CELEBRITIES, samples_per_speaker=samples)

    missing = [n for n, recs in records.items() if not recs]
    if missing:
        print(
            f"\n[benchmark] ERROR: no audio found for: {missing}\n"
            f"  Run with --download after filling in CELEBRITY_MANIFEST in\n"
            f"  data_pipeline/celebrity_loader.py, or place wav files in\n"
            f"  {DATA_DIR}/<name>/*.wav\n"
        )
        sys.exit(1)

    scenario = build_celebrity_scenario(records, forget_speaker=FORGET_CELEBRITY)
    print(
        f"[benchmark] scenario: "
        f"forget_train={len(scenario.forget_train)} "
        f"forget_test={len(scenario.forget_test)} "
        f"retain_train={len(scenario.retain_train)} "
        f"retain_test={len(scenario.retain_test)} "
        f"mia_holdout={len(scenario.mia_holdout)}"
    )

    # ---- DataLoaders ----
    train_loader = make_loader(scenario.original_train, shuffle=True)
    forget_train_loader = make_loader(scenario.forget_train, shuffle=True)
    forget_test_loader = make_loader(scenario.forget_test, shuffle=False)
    retain_train_loader = make_loader(scenario.retain_train, shuffle=True)
    retain_test_loader = make_loader(scenario.retain_test, shuffle=False)
    full_test_loader = make_loader(scenario.full_test, shuffle=False)

    train_epochs = 5 if args.quick else TRAIN_EPOCHS
    unlearn_epochs = 3 if args.quick else UNLEARN_EPOCHS
    eer_pairs = 200 if args.quick else 1000
    label_only_n_aug = 3 if args.quick else 10

    # ---- Train original model ----
    print("\n[benchmark] === Training original model ===")
    original_model = TinySpeakerCNN(num_classes=NUM_CLASSES)
    train_supervised(original_model, train_loader, full_test_loader, device, "original", train_epochs)

    # ---- Train oracle (retrain without forget) ----
    print("\n[benchmark] === Training oracle (retrain without forget data) ===")
    oracle_model = TinySpeakerCNN(num_classes=NUM_CLASSES)
    train_supervised(oracle_model, retain_train_loader, retain_test_loader, device, "oracle", train_epochs)

    # ---- Train SISA shards ----
    print("\n[benchmark] === Training SISA shards ===")
    all_records_flat = scenario.original_train
    sisa_models = train_sisa_shards(
        model_class=TinySpeakerCNN,
        all_records=all_records_flat,
        forget_records=scenario.forget_train,
        num_shards=SISA_SHARDS,
        epochs=train_epochs,
        lr=TRAIN_LR,
        batch_size=BATCH_SIZE,
        device=device,
        num_classes=NUM_CLASSES,
    )

    # ---- Unlearning config ----
    base_config = {
        "device": device,
        "lr": UNLEARN_LR,
        "epochs": unlearn_epochs,
        "num_classes": NUM_CLASSES,
    }
    scrub_config = {
        **base_config,
        "teacher_model": clone_model(original_model),
        "alpha": 1.0,
        "temperature": 4.0,
    }
    ssd_config = {
        **base_config,
        "ssd_alpha": 1.0,
        "ssd_eps": 1e-8,
        "ssd_n": 200,
    }

    method_configs = {
        "gradient_ascent": base_config,
        "random_label": base_config,
        "fine_tune": base_config,
        "scrub": scrub_config,
        "ssd": ssd_config,
    }

    # ---- Apply methods ----
    unlearned_models: Dict[str, nn.Module] = {}
    for method_name, config in method_configs.items():
        print(f"\n[benchmark] === Applying {method_name} ===")
        model_copy = clone_model(original_model)
        fn = REGISTRY[method_name]
        unlearned_models[method_name] = fn(model_copy, forget_train_loader, retain_train_loader, config)

    sisa_ensemble = SISAEnsemble(sisa_models).to(device)

    # ---- Evaluate all ----
    def _eval(model):
        return evaluate_all(
            model,
            forget_test_loader,
            retain_test_loader,
            full_test_loader,
            scenario.forget_train,
            scenario.mia_holdout,
            scenario.full_test,
            oracle_model,
            device,
            eer_pairs=eer_pairs,
            label_only_n_aug=label_only_n_aug,
        )

    rows = []

    print("\n[benchmark] === Evaluating ===")
    rows.append({"Method": "No Unlearning", **_eval(original_model)})
    rows.append({"Method": "Retrain Oracle", **_eval(oracle_model)})

    for method_name, model in unlearned_models.items():
        label = method_name.replace("_", " ").title()
        rows.append({"Method": label, **_eval(model)})

    rows.append({"Method": "SISA", **_eval(sisa_ensemble)})

    df = pd.DataFrame(rows).set_index("Method")
    print("\n=== Results ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    csv_path = ARTIFACT_DIR / "results_summary.csv"
    df.to_csv(csv_path)
    print(f"\n[benchmark] results saved to {csv_path}")

    # ---- Plots ----
    _save_results_plot(df, ARTIFACT_DIR / "results.png")
    _save_model_checkpoints(
        original_model,
        oracle_model,
        unlearned_models,
        sisa_models,
        MODEL_DIR,
    )
    _save_parameter_change_plots(
        original_model,
        {"Oracle": oracle_model, **{k.replace("_", " ").title(): v for k, v in unlearned_models.items()}},
        ARTIFACT_DIR,
    )

    try:
        models_dict = {
            "No Unlearning": original_model,
            "Oracle": oracle_model,
            **{k.replace("_", " ").title(): v for k, v in unlearned_models.items()},
            "SISA": sisa_ensemble,
        }
        plot_before_after_tsne_grid(
            models_dict,
            original_model,
            scenario.full_test,
            device,
            ARTIFACT_DIR / "before_after_tsne.png",
            celebrity_names=ALL_CELEBRITIES,
        )
    except Exception as exc:
        print(f"[benchmark] WARN: could not save t-SNE plot: {exc}")

def _save_results_plot(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        metrics = list(df.columns)
        n_cols = min(4, len(metrics))
        n_rows = math.ceil(len(metrics) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols)
        colors = plt.cm.tab10.colors

        for ax, metric in zip(axes.flat, metrics):
            vals = df[metric].values
            methods = df.index.tolist()
            bars = ax.bar(methods, vals, color=colors[: len(methods)])
            ax.set_title(metric)
            finite_vals = [float(v) for v in vals if np.isfinite(v)]
            max_val = max(finite_vals, default=1.0)
            ax.set_ylim(0, max(1.05, max_val * 1.1))
            ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
            for bar, val in zip(bars, vals):
                if not np.isfinite(val):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        for ax in axes.flat[len(metrics):]:
            ax.axis("off")

        fig.suptitle("Celebrity Speaker Unlearning Benchmark", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[benchmark] plot saved to {out_path}")
    except Exception as exc:
        print(f"[benchmark] WARN: could not save results plot: {exc}")


def _save_model_checkpoints(
    original_model: nn.Module,
    oracle_model: nn.Module,
    unlearned_models: Dict[str, nn.Module],
    sisa_models: List[nn.Module],
    out_dir: Path,
) -> None:
    checkpoints = {
        "original.pt": _checkpoint_payload(original_model),
        "oracle.pt": _checkpoint_payload(oracle_model),
    }

    for method_name, model in unlearned_models.items():
        checkpoints[f"{method_name}.pt"] = _checkpoint_payload(model)

    checkpoints["sisa.pt"] = {
        "model_type": "audio_sisa_ensemble",
        "num_classes": NUM_CLASSES,
        "label_names": ALL_CELEBRITIES,
        "num_shards": len(sisa_models),
        "shard_state_dicts": [
            {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            for model in sisa_models
        ],
    }

    for filename, payload in checkpoints.items():
        path = out_dir / filename
        torch.save(payload, path)
        print(f"[benchmark] checkpoint saved to {path}")


def _save_parameter_change_plots(
    original_model: nn.Module,
    comparison_models: Dict[str, nn.Module],
    out_dir: Path,
) -> None:
    original_state = {
        name: tensor.detach().cpu()
        for name, tensor in original_model.named_parameters()
    }

    for label, model in comparison_models.items():
        safe_label = label.lower().replace(" ", "_")
        out_path = out_dir / f"parameter_change_{safe_label}.png"
        plot_parameter_change_heatmap(
            original_state,
            {name: tensor.detach().cpu() for name, tensor in model.named_parameters()},
            out_path,
            title=f"Parameter Change vs Original: {label}",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Celebrity speaker unlearning benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 5 epochs, 10 clips per speaker",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download celebrity clips via yt-dlp before running",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(_parse_args())
