"""Membership Inference Attack (MIA) evaluators.

loss_threshold_mia  — loss-based threshold attack (refactored from benchmark.py)
label_only_mia      — label-only attack using prediction stability under noise
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data_pipeline.celebrity_loader import RecordDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_per_sample_losses(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: List[torch.Tensor] = []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            losses.append(criterion(model(features), labels).cpu())
    return torch.cat(losses).numpy()


def _make_loader(records, batch_size: int = 32) -> DataLoader:
    return DataLoader(RecordDataset(records), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Loss-threshold MIA
# ---------------------------------------------------------------------------

def loss_threshold_mia(
    model: nn.Module,
    forget_records: Sequence,
    holdout_records: Sequence,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Loss-threshold membership inference attack.

    Members (forget_records) should have lower loss than non-members (holdout_records)
    if the model has memorised the forget set.  AUC near 1.0 = high memorisation;
    AUC near 0.5 = successful unlearning.

    Returns
    -------
    Dict with keys: mia_auc, member_mean_loss, nonmember_mean_loss
    """
    forget_loader = _make_loader(forget_records, batch_size)
    holdout_loader = _make_loader(holdout_records, batch_size)

    member_losses = _collect_per_sample_losses(model, forget_loader, device)
    non_member_losses = _collect_per_sample_losses(model, holdout_loader, device)

    labels = np.concatenate([
        np.ones(len(member_losses), dtype=np.int64),
        np.zeros(len(non_member_losses), dtype=np.int64),
    ])
    # Lower loss = more likely member, so negate for AUC
    scores = np.concatenate([-member_losses, -non_member_losses])

    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")

    return {
        "mia_auc": auc,
        "member_mean_loss": float(member_losses.mean()),
        "nonmember_mean_loss": float(non_member_losses.mean()),
    }


# ---------------------------------------------------------------------------
# Label-only MIA
# ---------------------------------------------------------------------------

def label_only_mia(
    model: nn.Module,
    forget_records: Sequence,
    holdout_records: Sequence,
    device: torch.device,
    n_aug: int = 10,
    noise_std: float = 0.1,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Label-only MIA based on prediction stability under Gaussian MFCC noise.

    Members (training data) should produce more stable predictions under
    perturbation than non-members, because the model has over-fit to them.

    Stability score = fraction of augmented copies predicted as the original label.

    Returns
    -------
    Dict with keys: mia_auc, member_mean_stability, nonmember_mean_stability
    """
    def _stability_scores(records) -> np.ndarray:
        scores: List[float] = []
        model.eval()

        for record in records:
            base_features = record.features.unsqueeze(0).to(device)  # [1, 1, 40, 125]
            with torch.no_grad():
                original_pred = model(base_features).argmax(dim=1).item()

            correct_under_noise = 0
            for _ in range(n_aug):
                noisy = base_features + torch.randn_like(base_features) * noise_std
                with torch.no_grad():
                    pred = model(noisy).argmax(dim=1).item()
                if pred == original_pred:
                    correct_under_noise += 1

            scores.append(correct_under_noise / n_aug)

        return np.array(scores)

    member_stability = _stability_scores(forget_records)
    non_member_stability = _stability_scores(holdout_records)

    labels = np.concatenate([
        np.ones(len(member_stability), dtype=np.int64),
        np.zeros(len(non_member_stability), dtype=np.int64),
    ])
    scores = np.concatenate([member_stability, non_member_stability])

    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")

    return {
        "mia_auc": auc,
        "member_mean_stability": float(member_stability.mean()),
        "nonmember_mean_stability": float(non_member_stability.mean()),
    }
