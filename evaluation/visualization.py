"""Visualization utilities for the speech unlearning benchmark.

plot_tsne_embeddings          — t-SNE scatter of speaker embeddings
plot_before_after_tsne_grid   — grid: methods (cols) x before/after (rows)
plot_parameter_change_heatmap — per-layer L2 delta heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embeddings(
    model: nn.Module,
    records: Sequence,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (embeddings [N, D], labels [N], names [N])."""
    model.eval()
    embs: List[np.ndarray] = []
    labs: List[int] = []
    names: List[str] = []

    features_list = [r.features for r in records]
    labels_list = [r.label for r in records]
    name_list = [getattr(r, "celebrity_name", str(r.label)) for r in records]

    with torch.no_grad():
        for start in range(0, len(features_list), batch_size):
            batch = torch.stack(features_list[start: start + batch_size]).to(device)
            if hasattr(model, "embed"):
                emb = model.embed(batch).cpu().numpy()
            else:
                emb = model(batch).cpu().numpy()
            embs.append(emb)
            labs.extend(labels_list[start: start + batch_size])
            names.extend(name_list[start: start + batch_size])

    return np.concatenate(embs, axis=0), np.array(labs), names


def _tsne_reduce(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    perplexity = min(30, max(5, len(embeddings) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=500)
    return tsne.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Single t-SNE plot
# ---------------------------------------------------------------------------

def plot_tsne_embeddings(
    model: nn.Module,
    records: Sequence,
    device: torch.device,
    out_path: Path,
    celebrity_names: Optional[List[str]] = None,
    forget_name: str = "trump",
    title: str = "Speaker Embeddings (t-SNE)",
) -> None:
    """Plot t-SNE of speaker embeddings with forget speaker highlighted in red."""
    embeddings, labels, names = _get_embeddings(model, records, device)
    reduced = _tsne_reduce(embeddings)

    unique_names = celebrity_names or sorted(set(names))
    color_map = {}
    palette = plt.cm.tab10.colors
    for i, name in enumerate(unique_names):
        color_map[name] = "red" if name == forget_name else palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in unique_names:
        mask = np.array([n == name for n in names])
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[color_map[name]],
            label=name,
            alpha=0.75,
            s=40,
            edgecolors="k" if name == forget_name else "none",
            linewidths=0.5,
        )

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] t-SNE saved to {out_path}")


# ---------------------------------------------------------------------------
# Before / after grid
# ---------------------------------------------------------------------------

def plot_before_after_tsne_grid(
    models_dict: Dict[str, nn.Module],
    original_model: nn.Module,
    records: Sequence,
    device: torch.device,
    out_path: Path,
    celebrity_names: Optional[List[str]] = None,
    forget_name: str = "trump",
) -> None:
    """2-row grid: row 0 = before unlearning, row 1 = after.  One column per method.

    models_dict maps method_label -> unlearned model.
    original_model is the baseline (same for every "before" cell).
    """
    method_labels = list(models_dict.keys())
    n_methods = len(method_labels)

    unique_names = celebrity_names or sorted({getattr(r, "celebrity_name", str(r.label)) for r in records})
    palette = plt.cm.tab10.colors
    color_map = {
        name: "red" if name == forget_name else palette[i % len(palette)]
        for i, name in enumerate(unique_names)
    }

    fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Before", "After"]
    row_models = [original_model, None]  # None = filled per column

    for col, label in enumerate(method_labels):
        for row in range(2):
            ax = axes[row, col]
            model = original_model if row == 0 else models_dict[label]

            embeddings, _, names = _get_embeddings(model, records, device)
            reduced = _tsne_reduce(embeddings)

            for name in unique_names:
                mask = np.array([n == name for n in names])
                ax.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    c=[color_map[name]],
                    alpha=0.7,
                    s=30,
                    edgecolors="k" if name == forget_name else "none",
                    linewidths=0.4,
                )

            if row == 0:
                ax.set_title(label, fontsize=9)
            ax.set_ylabel(row_labels[row], fontsize=8)
            ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=color_map[n], label=n) for n in unique_names]
    fig.legend(handles=patches, loc="lower center", ncol=len(unique_names), fontsize=8)
    fig.suptitle("Speaker Embeddings: Before vs After Unlearning", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] before/after t-SNE grid saved to {out_path}")


# ---------------------------------------------------------------------------
# Parameter change heatmap
# ---------------------------------------------------------------------------

def plot_parameter_change_heatmap(
    original_state: Dict[str, torch.Tensor],
    unlearned_state: Dict[str, torch.Tensor],
    out_path: Path,
    title: str = "Parameter Change (L2 per layer)",
) -> None:
    """Bar chart of per-layer L2 norm delta between original and unlearned model."""
    layer_names: List[str] = []
    deltas: List[float] = []

    for name in original_state:
        if name in unlearned_state:
            delta = float(torch.norm(original_state[name].cpu() - unlearned_state[name].cpu()).item())
            layer_names.append(name)
            deltas.append(delta)

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.6), 4))
    ax.bar(range(len(layer_names)), deltas, color="steelblue")
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("L2 norm of change")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] parameter change heatmap saved to {out_path}")
