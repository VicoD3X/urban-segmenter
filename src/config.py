from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "test"
DEFAULT_MASKS_DIR = PROJECT_ROOT / "data" / "processed" / "masks" / "test"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unet_effnetv2b0.keras"
DEFAULT_API_URL = ""
DEFAULT_PREDICTION_BACKEND = "local"


@dataclass(frozen=True)
class AppConfig:
    """Configuration runtime de la démonstration Streamlit."""

    images_dir: Path
    masks_dir: Path
    api_url: str
    prediction_backend: str
    model_path: Path


def get_app_config() -> AppConfig:
    """Construit la configuration à partir des variables d'environnement."""
    return AppConfig(
        images_dir=Path(os.getenv("IMAGES_DIR", DEFAULT_IMAGES_DIR)),
        masks_dir=Path(os.getenv("MASKS_DIR", DEFAULT_MASKS_DIR)),
        api_url=os.getenv("API_URL", DEFAULT_API_URL).strip(),
        prediction_backend=os.getenv(
            "PREDICTION_BACKEND",
            DEFAULT_PREDICTION_BACKEND,
        ).strip().lower(),
        model_path=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)),
    )
