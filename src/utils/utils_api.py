from __future__ import annotations

import base64
import io
from typing import Any, Dict, Union

import numpy as np
import requests
from PIL import Image


def encode_image_to_base64(image_rgb: Union[np.ndarray, Image.Image]) -> str:
    """Encode une image RGB en base64 (format PNG).

    Paramètres
    ----------
    image_rgb : np.ndarray | PIL.Image.Image
        Image au format RGB. Les formats acceptés :
        - np.ndarray de forme (H, W, 3)
        - PIL.Image.Image (converti en RGB)

    Retour
    ------
    str
        Chaîne base64 correspondant à un PNG.
    """
    # Support des entrées PIL
    if isinstance(image_rgb, Image.Image):
        image_rgb = np.array(image_rgb.convert("RGB"), dtype=np.uint8)

    # Validation du format d'entrée (np.ndarray attendu après conversion)
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"image_rgb doit être un np.ndarray ou une PIL.Image. Reçu : {type(image_rgb)}")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb doit être de forme (H, W, 3). Reçu : {image_rgb.shape}")

    # Assure uint8 pour PNG (si jamais un float est fourni)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(image_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_mask_from_api(mask_data: Any) -> np.ndarray:
    """Convertit une structure JSON (liste de listes) en masque numpy 2D uint8."""
    mask = np.array(mask_data, dtype=np.uint8)

    if mask.ndim != 2:
        raise ValueError(f"Le masque reconstruit doit être 2D (H, W). Reçu : {mask.shape}")

    return mask


def send_image_to_api(image_rgb: Union[np.ndarray, Image.Image], api_url: str, timeout: float = 30.0) -> np.ndarray:
    """Envoie une image encodée à l'API et récupère le masque prédit."""
    img_b64 = encode_image_to_base64(image_rgb)
    payload: Dict[str, Any] = {"image": img_b64}

    response = requests.post(api_url, json=payload, timeout=timeout)

    # Tentative de parsing JSON pour enrichir les messages d'erreur
    data: Any = None
    try:
        data = response.json()
    except Exception:
        data = None

    if response.status_code != 200:
        detail = data if isinstance(data, dict) else response.text
        raise RuntimeError(f"Erreur API ({response.status_code}) : {detail}")

    if not isinstance(data, dict):
        raise RuntimeError("Réponse API invalide : un objet JSON était attendu.")

    if "mask_pred" not in data:
        raise KeyError("Champ 'mask_pred' manquant dans la réponse API.")

    return decode_mask_from_api(data["mask_pred"])
