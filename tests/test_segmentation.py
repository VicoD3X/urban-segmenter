import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation import remap_mask


def test_remap_mask_cityscapes_to_eight_classes():
    mask = np.array(
        [
            [0, 7, 8, 11],
            [17, 21, 23, 33],
        ],
        dtype=np.uint8,
    )

    remapped = remap_mask(mask)

    expected = np.array(
        [
            [0, 1, 2, 3],
            [5, 6, 7, 7],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(remapped, expected)
