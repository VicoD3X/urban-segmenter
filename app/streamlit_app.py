from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src.config import get_app_config
from src.inference import (
    load_keras_model,
    predict_mask_local,
    predict_mask_with_backend,
)
from src.segmentation import remap_mask
from src.utils.utils_data import list_available_ids, load_image_and_mask
from src.visualization import colorize_mask, np_to_pil, plot_class_importance


# ------------------------------------------------------------
# Configuration des chemins
# ------------------------------------------------------------
CONFIG = get_app_config()
IMAGES_DIR = CONFIG.images_dir
MASKS_DIR = CONFIG.masks_dir
API_URL = CONFIG.api_url
PREDICTION_BACKEND = CONFIG.prediction_backend
MODEL_PATH = CONFIG.model_path


@st.cache_data
def get_available_ids():
    """Retourne la liste des IDs disponibles (mise en cache)."""
    return list_available_ids(IMAGES_DIR)


@st.cache_data
def get_image_and_mask(image_id: str):
    """Charge l’image RGB et le masque correspondant à un ID (mise en cache)."""
    return load_image_and_mask(image_id, IMAGES_DIR, MASKS_DIR)


# ------------------------------------------------------------
# Prédiction locale (utilisée si PREDICTION_BACKEND=local ou en fallback)
# ------------------------------------------------------------
@st.cache_resource
def load_local_model(model_path: str):
    """Charge le modèle Keras une seule fois par session Streamlit (cache_resource)."""
    return load_keras_model(model_path)


def predict_local(image_rgb, target_shape):
    """Applique l’inférence locale en conservant le cache Streamlit du modèle."""
    model = load_local_model(str(MODEL_PATH))
    return predict_mask_local(image_rgb, target_shape, model)


# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="Cityscapes Segmentation Lab",
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
st.title("Cityscapes Segmentation Lab")
st.caption("Urban Segmenter - démonstrateur Computer Vision pour scènes urbaines")

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
st.sidebar.write(f"Backend : `{PREDICTION_BACKEND}`")
st.sidebar.write(f"Modèle local : `{MODEL_PATH}`")
if API_URL:
    st.sidebar.code(API_URL, language="text")
else:
    st.sidebar.info("API distante non configurée. Le mode local est recommandé pour la démo.")

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

    # Prédiction : local (recommandé) ou API + fallback local automatique
    with st.spinner("Prédiction en cours…"):
        try:
            mask_pred, used_local_fallback = predict_mask_with_backend(
                image_rgb=image_rgb,
                target_shape=mask_true.shape,
                backend=PREDICTION_BACKEND,
                api_url=API_URL,
                local_predictor=predict_local,
            )
            if used_local_fallback:
                st.warning("API indisponible : bascule automatique en prédiction locale.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
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
