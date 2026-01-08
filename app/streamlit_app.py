from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Imports utilitaires internes au projet
from src.utils.utils_data import list_available_ids, load_image_and_mask
from src.utils.utils_api import send_image_to_api
from src.utils.utils_visual import colorize_mask

# ------------------------------------------------------------
# Mapping Cityscapes 34 -> 8 classes (version légère pour Streamlit)
# ------------------------------------------------------------
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

# Libellés des 8 classes (utile pour le graphique)
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


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Applique le mapping Cityscapes 34 classes -> 8 classes.
    Utilisé uniquement pour la visualisation du masque réel côté Streamlit.
    """
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for old_id, new_id in CITYSCAPES_34_TO_8.items():
        new_mask[mask == old_id] = new_id
    return new_mask


# ------------------------------------------------------------
# Configuration des chemins
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = Path(
    os.getenv(
        "IMAGES_DIR",
        PROJECT_ROOT / "data" / "processed" / "images" / "test",
    )
)

MASKS_DIR = Path(
    os.getenv(
        "MASKS_DIR",
        PROJECT_ROOT / "data" / "processed" / "masks" / "test",
    )
)

# URL de l’API (modifiable via variable d’environnement)
API_URL = os.getenv(
    "API_URL",
    "https://p8oc-api-6972f71da6e9.herokuapp.com/predict",
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convertit un tableau NumPy (H, W, 3) en image PIL."""
    return Image.fromarray(arr.astype(np.uint8))


def plot_class_importance(mask_true_8: np.ndarray, mask_pred: np.ndarray) -> plt.Figure:
    """
    Graphique simple "features importantes" :
    répartition des classes (en % de pixels) sur masque réel vs masque prédit.
    Accessibilité : noir & blanc + hachures, pas de dépendance à la couleur.
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
    ax.set_ylabel("% de pixels")
    ax.set_ylim(0, max(float(pct_true.max()), float(pct_pred.max())) * 1.25 + 1)
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.legend()

    # Annotations légères (lisibles)
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


@st.cache_data
def get_available_ids():
    """Retourne la liste des IDs disponibles (mise en cache)."""
    return list_available_ids(IMAGES_DIR)


@st.cache_data
def get_image_and_mask(image_id: str):
    """Charge l’image RGB et le masque correspondant à un ID (mise en cache)."""
    return load_image_and_mask(image_id, IMAGES_DIR, MASKS_DIR)


# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="Projet P8 - Segmentation Cityscapes",
    layout="wide",
)

st.title("🚗 Projet P8 – Segmentation de scènes urbaines")

st.markdown(
    """
Application de démonstration du modèle de segmentation entraîné sur Cityscapes.

**Workflow :**
1. Sélection d’un ID d’image.
2. Chargement de l’image RGB et du masque réel.
3. Envoi de l’image à l’API de segmentation.
4. Visualisation du masque prédit et comparaison avec le masque réel.
"""
)

# Sidebar : informations de configuration + mode d’affichage (accessibilité 200% zoom)
st.sidebar.header("Configuration")
st.sidebar.write(f"📁 Dossier images : `{IMAGES_DIR}`")
st.sidebar.write(f"📁 Dossier masques : `{MASKS_DIR}`")
st.sidebar.write(f"🌐 URL API : `{API_URL}`")

display_mode = st.sidebar.selectbox(
    "Mode d’affichage (accessibilité)",
    ["Onglets (recommandé à 200% zoom)", "Colonnes (écran large)"],
    index=0,
)

# ------------------------------------------------------------
# Sélection et traitement de l'image
# ------------------------------------------------------------
try:
    ids = get_available_ids()
except Exception as e:
    st.error(f"Impossible de lister les IDs dans `{IMAGES_DIR}` : {e}")
    st.stop()

if not ids:
    st.error(f"Aucune image détectée dans `{IMAGES_DIR}`.")
    st.stop()

selected_id = st.selectbox("Sélection de l’ID de l’image :", ids)

if st.button("Lancer la prédiction sur cet ID"):
    # Chargement image + masque réel
    with st.spinner("Chargement des données..."):
        try:
            image_rgb, mask_true = get_image_and_mask(selected_id)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données pour `{selected_id}` : {e}")
            st.stop()

    # Appel API de segmentation
    with st.spinner("Appel à l’API de segmentation..."):
        try:
            # Conservation de la logique simple : envoi du np.ndarray à l’API
            mask_pred = send_image_to_api(image_rgb, API_URL)
        except Exception as e:
            st.error(f"Erreur lors de l’appel API : {e}")
            st.stop()

    # Colorisation pour visualisation
    try:
        # Remapping 34 -> 8 classes pour le masque réel (labelIds bruts)
        mask_true_remap = remap_mask(mask_true)
        mask_true_color = colorize_mask(mask_true_remap)

        # Le masque prédit est déjà en 0..7 (sortie du modèle)
        mask_pred_color = colorize_mask(mask_pred)
    except Exception as e:
        st.error(f"Erreur lors de la colorisation des masques : {e}")
        st.stop()

    # --------------------------------------------------------
    # Affichage des résultats (ergonomie OK à 200% zoom)
    # --------------------------------------------------------
    if display_mode == "Onglets (recommandé à 200% zoom)":
        tab1, tab2, tab3 = st.tabs(["Image RGB", "Masque réel (remappé)", "Masque prédit"])

        with tab1:
            st.subheader("Image RGB")
            st.image(np_to_pil(image_rgb), use_container_width=True)

        with tab2:
            st.subheader("Masque réel (remappé)")
            st.image(np_to_pil(mask_true_color), use_container_width=True)

        with tab3:
            st.subheader("Masque prédit")
            st.image(np_to_pil(mask_pred_color), use_container_width=True)

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Image RGB")
            st.image(np_to_pil(image_rgb), use_container_width=True)

        with col2:
            st.subheader("Masque réel (remappé)")
            st.image(np_to_pil(mask_true_color), use_container_width=True)

        with col3:
            st.subheader("Masque prédit")
            st.image(np_to_pil(mask_pred_color), use_container_width=True)

    # --------------------------------------------------------
    # Graphique "features importantes" (simple + accessible)
    # --------------------------------------------------------
    st.subheader("Répartition des classes (importance) — Réel vs Prédit")
    fig = plot_class_importance(mask_true_remap, mask_pred)
    st.pyplot(fig, use_container_width=True)

    st.success("Prédiction terminée.")
else:
    st.info("Sélectionner un ID puis lancer la prédiction.")
