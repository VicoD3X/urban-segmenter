from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.segmentation import CLASS_NAMES
from src.utils.utils_visual import colorize_mask


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convertit un tableau NumPy RGB en image PIL."""
    return Image.fromarray(arr.astype(np.uint8))


def plot_class_importance(mask_true_8: np.ndarray, mask_pred: np.ndarray) -> plt.Figure:
    """
    Compare la répartition des classes entre masque réel et masque prédit.

    Le graphique conserve une lecture noir et blanc grâce aux hachures.
    """
    n = len(CLASS_NAMES)

    counts_true = np.bincount(mask_true_8.ravel(), minlength=n)
    counts_pred = np.bincount(mask_pred.ravel(), minlength=n)

    pct_true = counts_true / max(counts_true.sum(), 1) * 100.0
    pct_pred = counts_pred / max(counts_pred.sum(), 1) * 100.0

    x = np.arange(n)
    w = 0.42

    fig, ax = plt.subplots(figsize=(10, 4))

    bars_true = ax.bar(
        x - w / 2,
        pct_true,
        width=w,
        color="white",
        edgecolor="black",
        hatch="///",
        label="Réel",
    )
    bars_pred = ax.bar(
        x + w / 2,
        pct_pred,
        width=w,
        color="white",
        edgecolor="black",
        hatch="xx",
        label="Prédit",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_ylabel("Proportion de pixels (%)")
    ax.set_ylim(0, max(float(pct_true.max()), float(pct_pred.max())) * 1.25 + 1)
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.legend()

    for b in list(bars_true) + list(bars_pred):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.4,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig
