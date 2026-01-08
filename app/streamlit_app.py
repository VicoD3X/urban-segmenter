from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Utilitaires internes au projet
from src.utils.utils_data import list_available_ids, load_image_and_mask
from src.utils.utils_api import send_image_to_api
from src.utils.utils_visual import colorize_mask


# Configuration Streamlit (doit être appelée avant les autres commandes Streamlit)
st.set_page_config(
    page_title="P8 OC - Segmentation Cityscapes",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    "background",
    "road",
    "sidewalk",
    "building",
    "other const.",
    "object",
    "vegetation",
    "vehicle",
]

# Nombre total de classes finales
N_CLASSES = 8


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Applique le mapping Cityscapes 34 → 8 classes sur un masque."""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for old_id, new_id in CITYSCAPES_34_TO_8.items():
        new_mask[mask == old_id] = new_id
    return new_mask


# ============================================================
# Configuration des chemins et paramètres globaux
# ============================================================
def find_project_root(start: Path) -> Path:
    """
    Tente de retrouver la racine du projet en remontant jusqu'à trouver
    un dossier contenant (src) et (data). Fallback sur un parent plausible.
    """
    start_dir = start if start.is_dir() else start.parent
    for p in [start_dir] + list(start_dir.parents):
        if (p / "src").exists() and (p / "data").exists():
            return p
    # Fallback (garde le comportement proche de ton code initial)
    return start_dir.parents[1] if len(start_dir.parents) > 1 else start_dir


_env_root = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(_env_root) if _env_root else find_project_root(Path(__file__).resolve())

# Répertoires des images et masques (configurables via variables d’environnement)
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", PROJECT_ROOT / "data" / "processed" / "images" / "test"))
MASKS_DIR = Path(os.getenv("MASKS_DIR", PROJECT_ROOT / "data" / "processed" / "masks" / "test"))

# URL de l’API de prédiction
API_URL = os.getenv("API_URL", "https://p8oc-api-6972f71da6e9.herokuapp.com/predict")


# ============================================================
# Helpers d’interface Streamlit (accessibilité et lisibilité)
# ============================================================
def page_header(title: str, description: str) -> None:
    """Affiche le titre principal et la description de la page."""
    st.title(title)
    st.markdown(description)


def chart_block(title: str, fig: plt.Figure, conclusion: str) -> None:
    """Affiche un graphique avec un titre et une conclusion associée."""
    st.subheader(title)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    st.caption(f"Conclusion : {conclusion}")


def image_block(title: str, arr: np.ndarray, caption: str) -> None:
    """Affiche une image avec un titre et une légende."""
    st.subheader(title)
    st.image(Image.fromarray(arr.astype(np.uint8)), use_container_width=True)
    st.caption(caption)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Conversion d’un tableau NumPy en image PIL."""
    return Image.fromarray(arr.astype(np.uint8))


# ============================================================
# Chargement des données avec mise en cache Streamlit
# ============================================================
@st.cache_data
def get_available_ids():
    """Liste les identifiants d’images disponibles."""
    return list_available_ids(IMAGES_DIR)


@st.cache_data
def get_image_and_mask(image_id: str):
    """Charge une image RGB et son masque associé."""
    return load_image_and_mask(image_id, IMAGES_DIR, MASKS_DIR)


# ============================================================
# Calcul des métriques de segmentation
# ============================================================
def compute_metrics(mask_true_8: np.ndarray, mask_pred: np.ndarray) -> dict:
    """Calcule pixel acc, mIoU, mDice + valeurs par classe."""
    mask_true_8 = mask_true_8.astype(np.int32)
    mask_pred = mask_pred.astype(np.int32)

    # Pixel accuracy globale
    acc = float((mask_true_8 == mask_pred).mean())

    ious = []
    dices = []

    # Calcul des métriques par classe
    for c in range(N_CLASSES):
        t = mask_true_8 == c
        p = mask_pred == c

        inter = int(np.logical_and(t, p).sum())
        union = int(np.logical_or(t, p).sum())
        denom = int(t.sum() + p.sum())

        iou = (inter / union) if union > 0 else np.nan
        dice = (2 * inter / denom) if denom > 0 else np.nan

        ious.append(iou)
        dices.append(dice)

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
    """Distribution des pixels par classe (réel vs prédit)."""
    pct_true = counts_true / max(counts_true.sum(), 1) * 100.0
    pct_pred = counts_pred / max(counts_pred.sum(), 1) * 100.0

    x = np.arange(N_CLASSES)
    w = 0.42

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    bars1 = ax.bar(
        x - w / 2,
        pct_true,
        width=w,
        color="white",
        edgecolor="black",
        hatch="///",
        label="Réel",
    )
    bars2 = ax.bar(
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
    ax.set_ylabel("% pixels")
    ax.set_ylim(0, max(pct_true.max(), pct_pred.max()) * 1.25 + 1)
    ax.legend()

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
    """Visualisation de l’IoU par classe."""
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    x = np.arange(N_CLASSES)
    vals = np.nan_to_num(ious, nan=0.0)

    bars = ax.bar(x, vals, color="white", edgecolor="black", hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.0)

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


def fig_top_class_gaps(counts_true: np.ndarray, counts_pred: np.ndarray) -> tuple[plt.Figure, list[int]]:
    """Identification des classes avec les plus grands écarts de distribution."""
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


# ============================================================
# Robustesse : conversion du retour API en masque label 2D
# ============================================================
def ensure_label_mask_2d(mask_pred_raw: Any, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Convertit ce que renvoie send_image_to_api en ndarray 2D (H, W) d'IDs [0..7].
    - Supporte: PIL.Image, np.ndarray, liste imbriquée
    - Redimensionne au besoin en NEAREST pour préserver les labels
    - Force les valeurs hors [0..7] à 0 (background)
    """
    if isinstance(mask_pred_raw, Image.Image):
        arr = np.asarray(mask_pred_raw)
    else:
        arr = np.asarray(mask_pred_raw)

    if arr.ndim == 3:
        # (H, W, C) : extraction d'une couche de labels
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 3:
            # Cas fréquent : masque grayscale stocké en RGB (3 canaux identiques)
            if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
                arr = arr[:, :, 0]
            else:
                raise ValueError(
                    "Le masque prédit semble être une image RGB colorisée (pas des IDs de classes). "
                    "L’API doit renvoyer un masque 2D d’IDs (0..7) pour calculer les métriques."
                )
        else:
            raise ValueError(f"Masque prédit: nombre de canaux inattendu: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Masque prédit: dimensions inattendues: {arr.shape}")

    # Resize si nécessaire (préserve les labels)
    if arr.shape != target_shape:
        arr_img = Image.fromarray(arr.astype(np.uint8))
        arr_img = arr_img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
        arr = np.asarray(arr_img)

    arr = arr.astype(np.int32)

    # Nettoyage des valeurs hors classes
    invalid = (arr < 0) | (arr >= N_CLASSES)
    if invalid.any():
        arr = arr.copy()
        arr[invalid] = 0

    return arr


# ============================================================
# UI PRINCIPALE STREAMLIT
# ============================================================
def main() -> None:
    page_header(
        "Segmentation Cityscapes (8 classes)",
        "Sélectionne une image de test, appelle l’API, puis compare **image**, **mask réel** et **mask prédit**.\n\n"
        "Les graphiques utilisent des motifs N&B pour rester lisibles (accessibilité).",
    )

    # Diagnostics de chemins / variables (utile en déploiement)
    with st.expander("🔧 Diagnostics"):
        st.write("PROJECT_ROOT:", str(PROJECT_ROOT))
        st.write("IMAGES_DIR:", str(IMAGES_DIR), "| exists =", IMAGES_DIR.exists())
        st.write("MASKS_DIR:", str(MASKS_DIR), "| exists =", MASKS_DIR.exists())
        st.write("API_URL (défaut):", API_URL)

    # Pré-check dossiers
    if not IMAGES_DIR.exists():
        st.error(f"Dossier images introuvable: {IMAGES_DIR}")
        st.stop()

    ids = get_available_ids()
    if not ids:
        st.warning(f"Aucune image trouvée dans: {IMAGES_DIR}")
        st.stop()

    # Sidebar
    st.sidebar.header("Paramètres")
    image_id = st.sidebar.selectbox("ID image", ids)
    api_url = st.sidebar.text_input("URL API de prédiction", value=API_URL)
    run_pred = st.sidebar.button("Lancer la prédiction", type="primary")

    # Chargement image + mask réel
    image_rgb, mask_true = get_image_and_mask(image_id)
    mask_true_8 = remap_mask(mask_true)

    # Reset session_state si l’image (ou l’URL) change
    request_key = f"{api_url}__{image_id}"
    if st.session_state.get("request_key") != request_key:
        st.session_state["request_key"] = request_key
        st.session_state["mask_pred"] = None

    # Layout 3 colonnes
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        image_block("Image", image_rgb, f"ID: {image_id}")

    with col2:
        image_block(
            "Mask réel (8 classes)",
            colorize_mask(mask_true_8),
            "Masque ground-truth remappé 34 → 8 classes.",
        )

    # Appel API (sur clic)
    if run_pred:
        with st.spinner("Appel API en cours..."):
            try:
                # Correction: envoi du np.ndarray (image_rgb) à l'API, pas d'objet PIL.Image
                raw_pred = send_image_to_api(image_rgb, api_url)
                mask_pred = ensure_label_mask_2d(raw_pred, target_shape=mask_true_8.shape)
                st.session_state["mask_pred"] = mask_pred
            except Exception as e:
                st.session_state["mask_pred"] = None
                st.error(f"Erreur lors de l’appel API / parsing du mask : {e}")

    # Affichage mask prédit
    with col3:
        if st.session_state.get("mask_pred") is None:
            st.subheader("Mask prédit")
            st.info("Cliquer sur **« Lancer la prédiction »** pour afficher le mask prédit.")
            st.stop()

        mask_pred = st.session_state["mask_pred"]
        image_block("Mask prédit", colorize_mask(mask_pred), "Masque renvoyé par l’API (IDs 0..7).")

    # Métriques + tableaux + graphes
    st.divider()

    metrics = compute_metrics(mask_true_8, mask_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pixel accuracy", f"{metrics['pixel_accuracy']:.3f}")
    m2.metric("mIoU", f"{metrics['mean_iou']:.3f}")
    m3.metric("mDice", f"{metrics['mean_dice']:.3f}")

    df = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "IoU": metrics["per_class_iou"],
            "Dice": metrics["per_class_dice"],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Distributions
    counts_true = np.bincount(mask_true_8.ravel(), minlength=N_CLASSES)
    counts_pred = np.bincount(mask_pred.ravel(), minlength=N_CLASSES)

    chart_block(
        "Distribution des pixels par classe (réel vs prédit)",
        fig_class_distribution(counts_true, counts_pred),
        "Sur/sous-prédiction potentielle de certaines classes (attention aux classes rares).",
    )

    chart_block(
        "IoU par classe",
        fig_iou_per_class(metrics["per_class_iou"]),
        "Les classes à faible IoU sont celles qui posent le plus de difficulté au modèle.",
    )

    fig_gap, top_idx = fig_top_class_gaps(counts_true, counts_pred)
    top_txt = ", ".join([CLASS_NAMES[i] for i in top_idx])
    chart_block(
        "Top 5 écarts de distribution (réel vs prédit)",
        fig_gap,
        f"Plus gros écarts observés sur : {top_txt}.",
    )


# Point d'entrée
main()
