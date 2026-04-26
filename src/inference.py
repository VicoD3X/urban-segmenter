from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils.utils_api import send_image_to_api


def load_keras_model(model_path: str | Path) -> Any:
    """Charge un modèle Keras sans compiler les métriques d'entraînement."""
    import tensorflow as tf

    return tf.keras.models.load_model(str(model_path), compile=False)


def predict_mask_local(
    image_rgb: np.ndarray,
    target_shape: tuple[int, int],
    model: Any,
) -> np.ndarray:
    """
    Prédit un masque 2D localement avec un modèle Keras.

    La logique est volontairement identique à celle utilisée historiquement dans
    Streamlit : resize sur l'input du modèle, normalisation, argmax puis resize
    du masque au format cible.
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    h_in, w_in = input_shape[1], input_shape[2]
    if h_in is None or w_in is None:
        h_in, w_in = target_shape[0], target_shape[1]

    img = Image.fromarray(image_rgb.astype(np.uint8)).resize(
        (w_in, h_in),
        resample=Image.BILINEAR,
    )
    x = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.asarray(pred)

    if pred.ndim == 4:
        pred = pred[0]

    if pred.ndim == 3 and pred.shape[-1] > 1:
        mask = np.argmax(pred, axis=-1)
    elif pred.ndim == 3 and pred.shape[-1] == 1:
        mask = pred[..., 0]
    elif pred.ndim == 2:
        mask = pred
    else:
        raise ValueError(f"Sortie modèle inattendue : shape={pred.shape}")

    mask = np.asarray(mask).astype(np.int32)

    if mask.shape != target_shape:
        mask_img = Image.fromarray(mask.astype(np.uint8)).resize(
            (target_shape[1], target_shape[0]),
            resample=Image.NEAREST,
        )
        mask = np.asarray(mask_img).astype(np.int32)

    return np.clip(mask, 0, 7).astype(np.uint8)


def predict_mask_with_backend(
    image_rgb: np.ndarray,
    target_shape: tuple[int, int],
    backend: str,
    api_url: str,
    local_predictor: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
) -> tuple[np.ndarray, bool]:
    """
    Prédit via le backend demandé.

    Retourne le masque et un booléen indiquant si un fallback local a été utilisé.
    """
    if backend == "local":
        return local_predictor(image_rgb, target_shape), False

    try:
        return send_image_to_api(image_rgb, api_url), False
    except Exception:
        return local_predictor(image_rgb, target_shape), True
