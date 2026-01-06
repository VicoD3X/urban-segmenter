from __future__ import annotations

import base64
import io
from typing import Dict

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------------
# Chargement du modèle U-Net VGG16
# ------------------------------------------------------------
MODEL_PATH = "models/unet_effnetv2b0.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="P8OC Segmentation API")

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


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Resize + normalisation pour modèle U-Net."""
    H, W = 256, 512
    img = Image.fromarray(image_rgb).resize((W, H), resample=Image.BILINEAR)
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)




# ------------------------------------------------------------
# ROUTE /predict
# ------------------------------------------------------------
@app.post("/predict")
def predict(payload: PredictPayload):
    """Prend une image en base64, renvoie mask_pred en JSON."""
    # Décodage
    image_rgb = decode_base64_image(payload.image)

    # Prétraitement
    batch = preprocess(image_rgb)

    # Prédiction
    pred = model.predict(batch)[0]              # (H,W,classes)
    mask = np.argmax(pred, axis=-1)            # (H,W)

    # Conversion en liste JSON
    mask_list = mask.astype(int).tolist()

    return {"mask_pred": mask_list}
