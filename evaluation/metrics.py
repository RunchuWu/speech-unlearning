"""Evaluation metrics for the speech unlearning benchmark.

compute_eer           — Equal Error Rate from cosine-similarity speaker pairs
compute_weight_distance — per-layer Frobenius norm delta versus a reference model
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# EER (Equal Error Rate)
# ---------------------------------------------------------------------------

def _get_embeddings(
    model: nn.Module,
    records: Sequence,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, D], labels [N]) for all records."""
    model.eval()
    embs: List[np.ndarray] = []
    labs: List[int] = []

    # Process in mini-batches
    features_list = [r.features for r in records]
    labels_list = [r.label for r in records]

    with torch.no_grad():
        for start in range(0, len(features_list), batch_size):
            batch = torch.stack(features_list[start: start + batch_size]).to(device)
            if hasattr(model, "embed"):
                emb = model.embed(batch).cpu().numpy()
            else:
                # Fallback: use penultimate layer output
                emb = model(batch).cpu().numpy()
            embs.append(emb)
            labs.extend(labels_list[start: start + batch_size])

    return np.concatenate(embs, axis=0), np.array(labs)


def compute_eer(
    model: nn.Module,
    records: Sequence,
    device: torch.device,
    num_pairs: int = 1000,
    seed: int = 42,
) -> float:
    """Compute Equal Error Rate using cosine similarity of speaker embeddings.

    Randomly samples num_pairs genuine and impostor pairs and finds the
    threshold where FAR == FRR.

    Returns
    -------
    eer : float  (0.0 = perfect, 0.5 = chance)
    """
    rng = random.Random(seed)
    embeddings, labels = _get_embeddings(model, records, device)

    # Build index by label
    label_to_indices: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(int(lbl), []).append(idx)

    unique_labels = list(label_to_indices.keys())
    if len(unique_labels) < 2:
        return float("nan")

    genuine_scores: List[float] = []
    impostor_scores: List[float] = []

    for _ in range(num_pairs):
        # Genuine pair: same speaker
        lbl = rng.choice(unique_labels)
        if len(label_to_indices[lbl]) < 2:
            continue
        i, j = rng.sample(label_to_indices[lbl], 2)
        e1, e2 = embeddings[i], embeddings[j]
        score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        genuine_scores.append(score)

        # Impostor pair: different speakers
        lbl_a, lbl_b = rng.sample(unique_labels, 2)
        ia = rng.choice(label_to_indices[lbl_a])
        ib = rng.choice(label_to_indices[lbl_b])
        e1, e2 = embeddings[ia], embeddings[ib]
        score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        impostor_scores.append(score)

    if not genuine_scores or not impostor_scores:
        return float("nan")

    genuine_arr = np.array(genuine_scores)
    impostor_arr = np.array(impostor_scores)
    thresholds = np.linspace(
        min(genuine_arr.min(), impostor_arr.min()),
        max(genuine_arr.max(), impostor_arr.max()),
        500,
    )

    best_eer = 1.0
    for thresh in thresholds:
        far = float((impostor_arr >= thresh).mean())   # impostor accepted
        frr = float((genuine_arr < thresh).mean())     # genuine rejected
        eer_candidate = abs(far - frr)
        if eer_candidate < abs(best_eer - 0.5) * 2:
            best_eer = (far + frr) / 2.0

    return best_eer


# ---------------------------------------------------------------------------
# Weight distance
# ---------------------------------------------------------------------------

def compute_weight_distance(
    model_a: nn.Module,
    model_b: nn.Module,
) -> Dict[str, float]:
    """Compute per-layer Frobenius norm between two models' parameters.

    Useful for measuring how far an unlearned model drifted from the oracle.

    Returns
    -------
    Dict mapping parameter name -> Frobenius norm of the difference.
    """
    sd_a = {n: p.detach().cpu() for n, p in model_a.named_parameters()}
    sd_b = {n: p.detach().cpu() for n, p in model_b.named_parameters()}

    distances: Dict[str, float] = {}
    for name in sd_a:
        if name in sd_b:
            distances[name] = float(torch.norm(sd_a[name] - sd_b[name]).item())

    return distances
