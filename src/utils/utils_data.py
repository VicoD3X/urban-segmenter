from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def list_available_ids(images_dir: str | Path) -> List[str]:
    """Retourne la liste des IDs d'images disponibles à partir des noms Cityscapes."""
    images_dir = Path(images_dir)
    ids: List[str] = []

    for img_path in sorted(images_dir.glob("*_leftImg8bit.png")):
        name = img_path.name
        image_id = name.replace("_leftImg8bit.png", "")
        ids.append(image_id)

    return ids


def load_image_and_mask(
    image_id: str,
    images_dir: str | Path,
    masks_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Charge une image RGB et son masque Cityscapes (labelIds) à partir d'un ID."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    img_name = f"{image_id}_leftImg8bit.png"
    mask_name = f"{image_id}_gtFine_labelIds.png"

    img_path = images_dir / img_name
    mask_path = masks_dir / mask_name

    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque introuvable : {mask_path}")

    # Image RGB
    try:
        image_rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"Échec de lecture image : {img_path}") from e

    # Masque labelIds (doit finir en 2D)
    try:
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img)
    except Exception as e:
        raise RuntimeError(f"Échec de lecture masque : {mask_path}") from e

    # Si jamais le PNG est chargé en 3 canaux, on prend le premier
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return image_rgb, mask.astype(np.uint8)
