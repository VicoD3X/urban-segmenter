import numpy as np

from src.utils.utils_visual import colorize_mask


def test_colorize_mask():
    mask = np.zeros((32,32), dtype=np.uint8)
    rgb = colorize_mask(mask)
    assert rgb.shape == (32, 32, 3)
