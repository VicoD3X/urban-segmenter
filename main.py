from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from src.config import DEFAULT_MODEL_PATH
from src.inference import load_keras_model, predict_mask_local

# ------------------------------------------------------------
# Chargement du modèle U-Net
# ------------------------------------------------------------
MODEL_PATH = DEFAULT_MODEL_PATH
API_OUTPUT_SHAPE = (256, 512)
model = load_keras_model(MODEL_PATH)

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Urban Segmenter API")

# ------------------------------------------------------------
# Schéma d’entrée
# ------------------------------------------------------------
class PredictPayload(BaseModel):
    image: str   # image encodée en base64


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def decode_base64_image(image_b64: str) -> np.ndarray:
    """Décodage base64 → RGB np.ndarray"""
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


# ------------------------------------------------------------
# ROUTE /predict
# ------------------------------------------------------------
@app.post("/predict")
def predict(payload: PredictPayload):
    """Prend une image en base64, renvoie mask_pred en JSON."""
    # Décodage
    image_rgb = decode_base64_image(payload.image)

    # Prédiction
    mask = predict_mask_local(
        image_rgb=image_rgb,
        target_shape=API_OUTPUT_SHAPE,
        model=model,
    )

    # Conversion en liste JSON
    mask_list = mask.astype(int).tolist()

    return {"mask_pred": mask_list}
