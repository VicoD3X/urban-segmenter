from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Utilitaires internes au projet
from src.utils.utils_data import list_available_ids, load_image_and_mask
from src.utils.utils_api import send_image_to_api
from src.utils.utils_visual import colorize_mask


# ============================================================
# Mapping Cityscapes : 34 classes originales -> 8 classes cibles
# Utilisé pour l’affichage et l’évaluation du masque réel
# ============================================================
CITYSCAPES_34_TO_8 = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,          # background
    7: 1,                                              # road
    8: 2,                                              # sidewalk
    11: 3,                                             # building
    12: 4, 13: 4, 14: 4, 15: 4, 16: 4,                 # other construction
    17: 5, 18: 5, 19: 5, 20: 5,                        # object
    21: 6, 22: 6,                                      # vegetation
    23: 7, 24: 7, 25: 7, 26: 7, 27: 7, 28: 7,
    29: 7, 30: 7, 31: 7, 32: 7, 33: 7                  # vehicle
}

# Noms des classes finales (8 classes)
CLASS_NAMES = [
    "background", "road", "sidewalk", "building",
    "other const.", "object", "vegetation", "vehicle"
]

# Nombre total de classes finales
N_CLASSES = 8


def remap_mask(mask: np.ndarray) -> np.ndarray:
    # Applique le mapping Cityscapes 34 → 8 classes sur un masque
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for old_id, new_id in CITYSCAPES_34_TO_8.items():
        new_mask[mask == old_id] = new_id
    return new_mask


# ============================================================
# Configuration des chemins et paramètres globaux
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Répertoires des images et masques (configurables via variables d’environnement)
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", PROJECT_ROOT / "data" / "processed" / "images" / "test"))
MASKS_DIR = Path(os.getenv("MASKS_DIR", PROJECT_ROOT / "data" / "processed" / "masks" / "test"))

# URL de l’API de prédiction
API_URL = os.getenv("API_URL", "https://p8oc-api-6972f71da6e9.herokuapp.com/predict")


# ============================================================
# Helpers d’interface Streamlit (accessibilité et lisibilité)
# ============================================================
def page_header(title: str, description: str) -> None:
    # Affiche le titre principal et la description de la page
    st.title(title)
    st.markdown(description)


def chart_block(title: str, fig: plt.Figure, conclusion: str) -> None:
    # Affiche un graphique avec un titre et une conclusion associée
    st.subheader(title)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    st.caption(f"Conclusion : {conclusion}")


def image_block(title: str, arr: np.ndarray, caption: str) -> None:
    # Affiche une image avec un titre et une légende
    st.subheader(title)
    st.image(Image.fromarray(arr.astype(np.uint8)), use_container_width=True)
    st.caption(caption)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    # Conversion d’un tableau NumPy en image PIL
    return Image.fromarray(arr.astype(np.uint8))


# ============================================================
# Chargement des données avec mise en cache Streamlit
# ============================================================
@st.cache_data
def get_available_ids():
    # Liste les identifiants d’images disponibles
    return list_available_ids(IMAGES_DIR)


@st.cache_data
def get_image_and_mask(image_id: str):
    # Charge une image RGB et son masque associé
    return load_image_and_mask(image_id, IMAGES_DIR, MASKS_DIR)


# ============================================================
# Calcul des métriques de segmentation
# ============================================================
def compute_metrics(mask_true_8: np.ndarray, mask_pred: np.ndarray) -> dict:
    # Conversion explicite des types pour éviter les erreurs NumPy
    mask_true_8 = mask_true_8.astype(np.int32)
    mask_pred = mask_pred.astype(np.int32)

    # Pixel accuracy globale
    acc = float((mask_true_8 == mask_pred).mean())

    ious = []
    dices = []

    # Calcul des métriques par classe
    for c in range(N_CLASSES):
        t = (mask_true_8 == c)
        p = (mask_pred == c)

        inter = int(np.logical_and(t, p).sum())
        union = int(np.logical_or(t, p).sum())
        denom = int(t.sum() + p.sum())

        iou = (inter / union) if union > 0 else np.nan
        dice = (2 * inter / denom) if denom > 0 else np.nan

        ious.append(iou)
        dices.append(dice)

    # Moyennes sur les classes
    miou = float(np.nanmean(ious))
    mdice = float(np.nanmean(dices))

    return {
        "pixel_accuracy": acc,
        "mean_iou": miou,
        "mean_dice": mdice,
        "per_class_iou": np.array(ious, dtype=np.float32),
        "per_class_dice": np.array(dices, dtype=np.float32),
    }


# ============================================================
# Fonctions de visualisation (compatibles N&B)
# ============================================================
def fig_class_distribution(counts_true: np.ndarray, counts_pred: np.ndarray) -> plt.Figure:
    # Distribution des pixels par classe (réel vs prédit)
    pct_true = counts_true / max(counts_true.sum(), 1) * 100.0
    pct_pred = counts_pred / max(counts_pred.sum(), 1) * 100.0

    x = np.arange(N_CLASSES)
    w = 0.42

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    bars1 = ax.bar(
        x - w / 2, pct_true,
        width=w, color="white", edgecolor="black",
        hatch="///", label="Réel"
    )
    bars2 = ax.bar(
        x + w / 2, pct_pred,
        width=w, color="white", edgecolor="black",
        hatch="xx", label="Prédit"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_ylabel("% pixels")
    ax.set_ylim(0, max(pct_true.max(), pct_pred.max()) * 1.25 + 1)
    ax.legend()

    # Annotation des barres
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    return fig


def fig_iou_per_class(ious: np.ndarray) -> plt.Figure:
    # Visualisation de l’IoU par classe
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    x = np.arange(N_CLASSES)
    vals = np.nan_to_num(ious, nan=0.0)

    bars = ax.bar(x, vals, color="white", edgecolor="black", hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.0)

    # Annotation des barres
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    return fig


def fig_top_class_gaps(
    counts_true: np.ndarray,
    counts_pred: np.ndarray,
) -> tuple[plt.Figure, list[int]]:
    # Identification des classes avec les plus grands écarts de distribution
    pct_true = counts_true / max(counts_true.sum(), 1) * 100.0
    pct_pred = counts_pred / max(counts_pred.sum(), 1) * 100.0

    gaps = np.abs(pct_true - pct_pred)
    top_idx = np.argsort(gaps)[::-1][:5].tolist()

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    y = gaps[top_idx]
    labels = [CLASS_NAMES[i] for i in top_idx]

    bars = ax.bar(
        np.arange(len(top_idx)),
        y,
        color="white",
        edgecolor="black",
        hatch="\\\\",
    )

    ax.set_xticks(np.arange(len(top_idx)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Écart % (réel vs prédit)")
    ax.set_ylim(0, max(y) * 1.25 + 1)

    # Annotation des barres
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    return fig, top_idx
