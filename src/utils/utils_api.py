from __future__ import annotations

import base64
import io
from typing import Any, Dict

import numpy as np
import requests
from PIL import Image


def encode_image_to_base64(image_rgb: np.ndarray) -> str:
    """Encode une image RGB (H, W, 3) en base64 (format PNG)."""
    # Validation du format d'entrée
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb doit être de forme (H, W, 3). Reçu : {image_rgb.shape}")

    # Assure uint8 pour PNG (si jamais on reçoit du float)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(image_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_mask_from_api(mask_data: Any) -> np.ndarray:
    """Convertit une structure JSON (liste de listes) en masque numpy 2D uint8."""
    mask = np.array(mask_data, dtype=np.uint8)

    # Vérification structurelle
    if mask.ndim != 2:
        raise ValueError(f"Le masque reconstruit doit être 2D (H, W). Reçu : {mask.shape}")

    return mask


def send_image_to_api(image_rgb: np.ndarray, api_url: str) -> np.ndarray:
    """Envoie une image encodée à l'API et récupère le masque prédit."""
    img_b64 = encode_image_to_base64(image_rgb)
    payload: Dict[str, Any] = {"image": img_b64}

    response = requests.post(api_url, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Erreur API ({response.status_code}) : {response.text}")

    data = response.json()

    if "mask_pred" not in data:
        raise KeyError("Champ 'mask_pred' manquant dans la réponse API.")

    return decode_mask_from_api(data["mask_pred"])
