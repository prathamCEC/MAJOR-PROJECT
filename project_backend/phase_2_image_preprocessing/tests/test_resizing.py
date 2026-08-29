"""
Tests for Resizing and Padding Module.
"""

import numpy as np
import pytest

from phase_2_image_preprocessing.src.resizing import resize_with_aspect_ratio_and_pad


def test_resize_wide_image() -> None:
    """Test resizing a wide aspect ratio image (e.g. 100 x 400)."""
    wide_img = np.ones((100, 400), dtype=np.uint8) * 200
    resized = resize_with_aspect_ratio_and_pad(wide_img, target_size=(224, 224), pad_value=0)

    assert resized.shape == (224, 224)
    # The top and bottom should be padded with 0
    assert resized[0, 112] == 0
    assert resized[223, 112] == 0
    # The center row should have content (200)
    assert resized[112, 112] == 200


def test_resize_tall_image() -> None:
    """Test resizing a tall aspect ratio image (e.g. 500 x 200)."""
    tall_img = np.ones((500, 200), dtype=np.uint8) * 180
    resized = resize_with_aspect_ratio_and_pad(tall_img, target_size=(224, 224), pad_value=0)

    assert resized.shape == (224, 224)
    # The left and right borders should be padded with 0
    assert resized[112, 0] == 0
    assert resized[112, 223] == 0
    # Center column should have content
    assert resized[112, 112] == 180


def test_resize_square_image() -> None:
    """Test resizing a square image (e.g. 512 x 512)."""
    sq_img = np.ones((512, 512), dtype=np.uint8) * 150
    resized = resize_with_aspect_ratio_and_pad(sq_img, target_size=(224, 224), pad_value=0)

    assert resized.shape == (224, 224)
    # No zero padding should be present
    assert np.all(resized == 150)


def test_resize_color_image() -> None:
    """Test resizing 3-channel color image."""
    color_img = np.ones((300, 600, 3), dtype=np.uint8) * 120
    resized = resize_with_aspect_ratio_and_pad(
        color_img, target_size=(224, 224), pad_value=(0, 0, 0)
    )

    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8
