import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import inference


def test_predict_mask_with_backend_local_uses_local_predictor():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    expected = np.ones((2, 2), dtype=np.uint8)

    def local_predictor(image_rgb, target_shape):
        assert image_rgb.shape == image.shape
        assert target_shape == (2, 2)
        return expected

    mask, used_fallback = inference.predict_mask_with_backend(
        image_rgb=image,
        target_shape=(2, 2),
        backend="local",
        api_url="",
        local_predictor=local_predictor,
    )

    assert used_fallback is False
    assert np.array_equal(mask, expected)


def test_predict_mask_with_backend_api_success(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    expected = np.full((2, 2), 3, dtype=np.uint8)

    def fake_send_image_to_api(image_rgb, api_url):
        assert image_rgb.shape == image.shape
        assert api_url == "https://example.test/predict"
        return expected

    def local_predictor(_image_rgb, _target_shape):
        raise AssertionError("Le fallback local ne doit pas être appelé.")

    monkeypatch.setattr(inference, "send_image_to_api", fake_send_image_to_api)

    mask, used_fallback = inference.predict_mask_with_backend(
        image_rgb=image,
        target_shape=(2, 2),
        backend="api",
        api_url="https://example.test/predict",
        local_predictor=local_predictor,
    )

    assert used_fallback is False
    assert np.array_equal(mask, expected)


def test_predict_mask_with_backend_api_failure_falls_back(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    expected = np.full((2, 2), 7, dtype=np.uint8)

    def fake_send_image_to_api(_image_rgb, _api_url):
        raise RuntimeError("API indisponible")

    def local_predictor(image_rgb, target_shape):
        assert image_rgb.shape == image.shape
        assert target_shape == (2, 2)
        return expected

    monkeypatch.setattr(inference, "send_image_to_api", fake_send_image_to_api)

    mask, used_fallback = inference.predict_mask_with_backend(
        image_rgb=image,
        target_shape=(2, 2),
        backend="api",
        api_url="https://example.test/predict",
        local_predictor=local_predictor,
    )

    assert used_fallback is True
    assert np.array_equal(mask, expected)
