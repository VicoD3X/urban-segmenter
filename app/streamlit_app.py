from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

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

# Sidebar : informations de configuration
st.sidebar.header("Configuration")
st.sidebar.write(f"📁 Dossier images : `{IMAGES_DIR}`")
st.sidebar.write(f"📁 Dossier masques : `{MASKS_DIR}`")
st.sidebar.write(f"🌐 URL API : `{API_URL}`")

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
    # Affichage des résultats
    # --------------------------------------------------------
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

    st.success("Prédiction terminée.")
else:
    st.info("Sélectionner un ID puis lancer la prédiction.")
