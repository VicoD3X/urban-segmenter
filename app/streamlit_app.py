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
    page_title="P8 – Segmentation Cityscapes",
    layout="wide",
)

# Design : contraintes de lisibilité à 200% + hiérarchie typographique
st.markdown(
    """
<style>
/* Largeur lisible et stable (utile à 200% zoom) */
.block-container {
    max-width: 1200px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Typographie : lisibilité */
.stMarkdown, .stText, .stCaption, .stAlert, .stSelectbox, .stButton, .stRadio, .stCheckbox {
    font-size: 1.02rem;
    line-height: 1.55;
}

/* Hiérarchie titres */
h1 { margin-bottom: 0.25rem; }
h2 { margin-top: 1.6rem; }
h3 { margin-top: 1.2rem; }

/* Focus visible (clavier) */
button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid rgba(0,0,0,0.45) !important;
    outline-offset: 2px !important;
}

/* Sidebar : un peu d’air */
section[data-testid="stSidebar"] { padding-top: 1rem; }

/* Espacements des éléments */
div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# Header “produit”
st.title("Segmentation sémantique — scènes urbaines")
st.caption("Projet P8 (OpenClassrooms) • Démonstrateur de segmentation Cityscapes (8 classes)")

st.markdown(
    """
Cette application présente un flux complet : sélection d’une image, comparaison du masque réel et du masque prédit,
et analyse de la répartition des classes.

**Accessibilité :** l’interface reste lisible à 200% de zoom ; le graphique utilise des hachures (N&B) pour éviter une dépendance à la couleur.
"""
)

st.divider()

# Sidebar : organisation “pro”
st.sidebar.header("Paramètres")

display_mode = st.sidebar.selectbox(
    "Affichage des résultats",
    ["Onglets (recommandé à 200% zoom)", "Colonnes (écran large)"],
    index=0,
)

st.sidebar.markdown("### Sources")
images_ok = IMAGES_DIR.exists()
masks_ok = MASKS_DIR.exists()
st.sidebar.write(f"Images : {'✅' if images_ok else '❌'} `{IMAGES_DIR}`")
st.sidebar.write(f"Masques : {'✅' if masks_ok else '❌'} `{MASKS_DIR}`")

st.sidebar.markdown("### Service de prédiction")
st.sidebar.code(API_URL, language="text")

with st.sidebar.expander("Aide / informations"):
    st.write(
        "Le service de prédiction peut nécessiter un temps d’initialisation selon l’infrastructure d’hébergement. "
        "Un premier appel peut être plus lent."
    )
    st.write("Le graphique d’analyse reste accessible (hachures + contours).")

# Pré-check dossiers (message clair)
if not images_ok:
    st.error(f"Dossier images introuvable : {IMAGES_DIR}")
    st.stop()
if not masks_ok:
    st.error(f"Dossier masques introuvable : {MASKS_DIR}")
    st.stop()

# Sélection ID
try:
    ids = get_available_ids()
except Exception as e:
    st.error(f"Impossible de lister les IDs dans `{IMAGES_DIR}` : {e}")
    st.stop()

if not ids:
    st.error(f"Aucune image détectée dans `{IMAGES_DIR}`.")
    st.stop()

st.subheader("Sélection")
selected_id = st.selectbox("ID de l’image", ids)

st.divider()

# Action principale
if st.button("Lancer la prédiction", type="primary"):
    # Chargement image + masque réel
    with st.spinner("Chargement des données…"):
        try:
            image_rgb, mask_true = get_image_and_mask(selected_id)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données pour `{selected_id}` : {e}")
            st.stop()

    # Appel API de segmentation (logique inchangée)
    with st.spinner("Prédiction en cours…"):
        try:
            mask_pred = send_image_to_api(image_rgb, API_URL)
        except Exception as e:
            st.error(f"Erreur lors de l’appel API : {e}")
            st.stop()

    # Colorisation pour visualisation
    try:
        mask_true_remap = remap_mask(mask_true)
        mask_true_color = colorize_mask(mask_true_remap)
        mask_pred_color = colorize_mask(mask_pred)
    except Exception as e:
        st.error(f"Erreur lors de la colorisation des masques : {e}")
        st.stop()

    # Résumé rapide (vitrine)
    st.subheader("Résumé")
    c1, c2, c3 = st.columns(3)
    c1.metric("ID", selected_id)
    c2.metric("Taille image", f"{image_rgb.shape[1]} × {image_rgb.shape[0]}")
    c3.metric("Classes", "8")

    st.divider()

    st.subheader("Résultats")
    if display_mode == "Onglets (recommandé à 200% zoom)":
        tab1, tab2, tab3 = st.tabs(["Image RGB", "Masque réel (remappé)", "Masque prédit"])

        with tab1:
            st.image(np_to_pil(image_rgb), caption="Image d’entrée", use_container_width=True)

        with tab2:
            st.image(np_to_pil(mask_true_color), caption="Masque réel remappé (34 → 8)", use_container_width=True)

        with tab3:
            st.image(np_to_pil(mask_pred_color), caption="Masque prédit (0 → 7)", use_container_width=True)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(np_to_pil(image_rgb), caption="Image d’entrée", use_container_width=True)
        with col2:
            st.image(np_to_pil(mask_true_color), caption="Masque réel remappé (34 → 8)", use_container_width=True)
        with col3:
            st.image(np_to_pil(mask_pred_color), caption="Masque prédit (0 → 7)", use_container_width=True)

    st.divider()

    st.subheader("Analyse — répartition des classes (importance)")
    fig = plot_class_importance(mask_true_remap, mask_pred)
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)

    st.success("Prédiction terminée.")
else:
    st.info("Sélectionner un ID, puis lancer la prédiction.")
