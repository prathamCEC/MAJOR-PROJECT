"""
Tests for Normalization Module.
"""

import numpy as np
import pytest

from phase_2_image_preprocessing.src.normalization import (
    normalize_to_float32,
    convert_to_uint8,
)


def test_normalize_to_float32() -> None:
    """Test uint8 to float32 [0.0, 1.0] scaling."""
    img_uint8 = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    norm = normalize_to_float32(img_uint8, target_min=0.0, target_max=1.0)

    assert norm.dtype == np.float32
    assert norm.shape == (2, 2)
    assert pytest.approx(float(norm[0, 0]), 0.001) == 0.0
    assert pytest.approx(float(norm[1, 0]), 0.001) == 1.0
    assert pytest.approx(float(norm[0, 1]), 0.01) == 128 / 255.0


def test_convert_to_uint8_from_float() -> None:
    """Test float32 [0.0, 1.0] scaling to uint8 [0, 255]."""
    img_float = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    img_uint8 = convert_to_uint8(img_float, source_min=0.0, source_max=1.0)

    assert img_uint8.dtype == np.uint8
    assert img_uint8[0, 0] == 0
    assert img_uint8[1, 0] == 255
    assert img_uint8[0, 1] in (127, 128)
    assert img_uint8[1, 1] in (63, 64)


def test_roundtrip_normalization_preserves_range() -> None:
    """Test that uint8 -> float32 -> uint8 roundtrip avoids dark image bug."""
    original = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
    norm = normalize_to_float32(original)
    recovered = convert_to_uint8(norm)

    assert recovered.dtype == np.uint8
    # Max difference between roundtrip should be at most 1 due to rounding
    diff = np.abs(original.astype(int) - recovered.astype(int))
    assert np.max(diff) <= 1
