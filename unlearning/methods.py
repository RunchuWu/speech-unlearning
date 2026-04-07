"""All unlearning methods for the speech unlearning benchmark.

Methods
-------
gradient_ascent  (GA)   — ascend on forget loss, descend on retain loss
random_label     (RL)   — train forget data toward wrong classes, repair on retain
fine_tune        (FT)   — train on retain data only; ignore forget data entirely
scrub            (SCRUB) — KL-divergence against frozen teacher for forget/retain
ssd              (SSD)  — selective synaptic dampening via Fisher diagonal
sisa             (SISA) — sharded, isolated, sliced, aggregated retraining
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sample guaranteed incorrect labels (never equal to the true class)."""
    wrong = torch.randint(0, num_classes - 1, labels.shape, device=labels.device)
    return wrong + (wrong >= labels).long()


# ---------------------------------------------------------------------------
# Method 1: Gradient Ascent (GA)
# ---------------------------------------------------------------------------

def unlearn_gradient_ascent(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict,
) -> nn.Module:
    """Ascend on forget loss to erase signal, then descend on retain loss to repair."""
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    model.to(device)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        forget_obj = 0.0
        forget_n = 0
        retain_loss = 0.0
        retain_n = 0

        for features, labels in forget_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            (-loss).backward()
            optimizer.step()
            forget_obj += loss.item() * labels.size(0)
            forget_n += labels.size(0)

        for features, labels in retain_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            retain_loss += loss.item() * labels.size(0)
            retain_n += labels.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            print(
                f"[GA] epoch {epoch:02d}/{config['epochs']} "
                f"forget_ce={forget_obj/forget_n:.4f} "
                f"retain_ce={retain_loss/retain_n:.4f}"
            )

    return model


# ---------------------------------------------------------------------------
# Method 2: Random Label (RL)
# ---------------------------------------------------------------------------

def unlearn_random_label(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict,
) -> nn.Module:
    """Train forget samples toward wrong classes, then repair retain performance."""
    device = config["device"]
    num_classes = int(config["num_classes"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.to(device)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        forget_loss = 0.0
        forget_n = 0
        retain_loss = 0.0
        retain_n = 0

        for features, labels in forget_loader:
            features, labels = features.to(device), labels.to(device)
            wrong = _make_wrong_labels(labels, num_classes)
            optimizer.zero_grad()
            loss = criterion(model(features), wrong)
            loss.backward()
            optimizer.step()
            forget_loss += loss.item() * labels.size(0)
            forget_n += labels.size(0)

        for features, labels in retain_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            retain_loss += loss.item() * labels.size(0)
            retain_n += labels.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            print(
                f"[RL] epoch {epoch:02d}/{config['epochs']} "
                f"forget_wrong_ce={forget_loss/forget_n:.4f} "
                f"retain_ce={retain_loss/retain_n:.4f}"
            )

    return model


# ---------------------------------------------------------------------------
# Method 3: Fine-Tune Only (FT)
# ---------------------------------------------------------------------------

def unlearn_fine_tune(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict,
) -> nn.Module:
    """Fine-tune on retain data only — never touches forget data."""
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.to(device)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        retain_loss = 0.0
        retain_n = 0

        for features, labels in retain_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            retain_loss += loss.item() * labels.size(0)
            retain_n += labels.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            print(
                f"[FT] epoch {epoch:02d}/{config['epochs']} "
                f"retain_ce={retain_loss/retain_n:.4f}"
            )

    return model


# ---------------------------------------------------------------------------
# Method 4: SCRUB
# ---------------------------------------------------------------------------

def unlearn_scrub(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict,
) -> nn.Module:
    """SCRUB: maximize KL divergence from teacher on forget set, minimize on retain set.

    config must include:
        teacher_model  — frozen copy of the original trained model
        alpha          — weight on the retain KL term (default 1.0)
        temperature    — softmax temperature for KL (default 4.0)
    """
    device = config["device"]
    teacher: nn.Module = config["teacher_model"]
    alpha = float(config.get("alpha", 1.0))
    T = float(config.get("temperature", 4.0))

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        forget_kl = 0.0
        forget_n = 0
        retain_kl = 0.0
        retain_n = 0

        # Forget phase: maximize KL(student || teacher) — push student away
        for features, _ in forget_loader:
            features = features.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(features) / T
            s_logits = model(features) / T
            kl = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )
            (-kl).backward()  # ascent: maximize divergence on forget set
            optimizer.step()
            forget_kl += kl.item() * features.size(0)
            forget_n += features.size(0)

        # Retain phase: minimize KL(student || teacher) — stay close on retain set
        for features, labels in retain_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(features) / T
            s_logits = model(features) / T
            kl = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )
            ce = F.cross_entropy(model(features), labels)
            loss = alpha * kl + ce
            loss.backward()
            optimizer.step()
            retain_kl += kl.item() * features.size(0)
            retain_n += features.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            print(
                f"[SCRUB] epoch {epoch:02d}/{config['epochs']} "
                f"forget_kl={forget_kl/forget_n:.4f} "
                f"retain_kl={retain_kl/retain_n:.4f}"
            )

    return model


# ---------------------------------------------------------------------------
# Method 5: Selective Synaptic Dampening (SSD)
# ---------------------------------------------------------------------------

def _compute_fisher_diagonal(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """Estimate the diagonal Fisher information matrix via squared gradients."""
    model.eval()
    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad
    }
    criterion = nn.CrossEntropyLoss()
    total = 0

    for features, labels in loader:
        if total >= n_samples:
            break
        features, labels = features.to(device), labels.to(device)
        model.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach() ** 2 * labels.size(0)
        total += labels.size(0)

    for name in fisher:
        fisher[name] /= max(total, 1)

    return fisher


def unlearn_ssd(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: Dict,
) -> nn.Module:
    """Selective Synaptic Dampening: analytically dampen parameters important to forget set.

    config keys:
        ssd_alpha  — dampening strength (default 1.0)
        ssd_eps    — numerical stability (default 1e-8)
        ssd_n      — samples for Fisher estimate (default 200)
    """
    device = config["device"]
    alpha = float(config.get("ssd_alpha", 1.0))
    eps = float(config.get("ssd_eps", 1e-8))
    n = int(config.get("ssd_n", 200))

    model.to(device)
    print("[SSD] computing Fisher diagonals...")
    f_forget = _compute_fisher_diagonal(model, forget_loader, device, n)
    f_retain = _compute_fisher_diagonal(model, retain_loader, device, n)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ratio = f_forget[name] / (f_retain[name] + eps)
            p.data -= alpha * ratio * p.data

    print("[SSD] dampening complete.")
    return model


# ---------------------------------------------------------------------------
# Method 6: SISA (Sharded, Isolated, Sliced, Aggregated)
# ---------------------------------------------------------------------------

class _IndexDataset(Dataset):
    """Wrap a list of (features, label) tuples as a Dataset."""

    def __init__(self, records: List[Tuple[torch.Tensor, int]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        features, label = self.records[idx]
        return features, label


def train_sisa_shards(
    model_class,
    all_records: List,
    forget_records: List,
    num_shards: int = 4,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: torch.device = None,
    num_classes: int = 5,
) -> List[nn.Module]:
    """Train K shard sub-models, each excluding a random partition of the forget set.

    Each shard is trained on (all_retain ∪ shard_retain_slice) where
    shard_retain_slice ⊆ all_records \ forget_records.
    """
    if device is None:
        device = torch.device("cpu")

    forget_ids = {getattr(r, "sample_id", id(r)) for r in forget_records}
    retain_records = [r for r in all_records if getattr(r, "sample_id", id(r)) not in forget_ids]

    # Split retain records into shards
    random.shuffle(retain_records)
    shards = [retain_records[i::num_shards] for i in range(num_shards)]

    criterion = nn.CrossEntropyLoss()
    shard_models: List[nn.Module] = []

    for shard_idx, shard in enumerate(shards):
        print(f"[SISA] training shard {shard_idx + 1}/{num_shards} ({len(shard)} samples)")
        tuples = [(r.features, r.label) for r in shard]
        dataset = _IndexDataset(tuples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        m = model_class(num_classes=num_classes).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr)

        m.train()
        for epoch in range(epochs):
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                opt.zero_grad()
                loss = criterion(m(features), labels)
                loss.backward()
                opt.step()

        shard_models.append(m)

    return shard_models


def aggregate_sisa_predictions(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Average softmax predictions across all shard models."""
    all_probs = []
    for m in models:
        m.to(device)
        m.eval()
        probs = []
        with torch.no_grad():
            for features, _ in loader:
                features = features.to(device)
                p = F.softmax(m(features), dim=-1).cpu().numpy()
                probs.append(p)
        all_probs.append(np.concatenate(probs, axis=0))

    return np.mean(all_probs, axis=0)


def unlearn_sisa(
    shard_models: List[nn.Module],
    forget_records: List,
    config: Dict,
) -> List[nn.Module]:
    """Retrain only the shards that contained forget data (no-op here — shards were built forget-free).

    In this implementation shards are already trained without forget data, so
    unlearn_sisa is a no-op that returns the same list. The function exists so
    that the SISA evaluation path mirrors the interface of other methods.
    """
    print("[SISA] shards were trained without forget data — no retraining needed.")
    return shard_models


# ---------------------------------------------------------------------------
# Registry (SISA excluded — different interface)
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable] = {
    "gradient_ascent": unlearn_gradient_ascent,
    "random_label": unlearn_random_label,
    "fine_tune": unlearn_fine_tune,
    "scrub": unlearn_scrub,
    "ssd": unlearn_ssd,
}
