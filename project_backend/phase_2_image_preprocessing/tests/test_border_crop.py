"""
Tests for Border Cropping Module.
"""

import numpy as np
import pytest

from phase_2_image_preprocessing.src.border_crop import detect_and_crop_borders


def test_crop_image_with_black_border() -> None:
    """Test that a distinct artificial black border is correctly cropped."""
    # Create 100x100 image with 20px black borders around a 60x60 bright content center
    full_img = np.zeros((100, 100), dtype=np.uint8)
    full_img[20:80, 20:80] = 180

    cropped = detect_and_crop_borders(full_img, threshold=10, margin=2)

    # Content is 60x60; with margin of 2, dimensions should be around 64x64
    assert cropped.shape[0] < 100
    assert cropped.shape[1] < 100
    assert cropped.shape[0] >= 60
    assert cropped.shape[1] >= 60


def test_crop_image_without_border_preserved() -> None:
    """Test that an image with full content across all borders is untouched."""
    # Content everywhere
    full_img = np.ones((100, 100), dtype=np.uint8) * 150

    cropped = detect_and_crop_borders(full_img, threshold=10, margin=2)

    assert cropped.shape == (100, 100)
    assert np.array_equal(cropped, full_img)


def test_crop_color_fundus_with_border() -> None:
    """Test border crop on 3-channel color image."""
    full_img = np.zeros((120, 120, 3), dtype=np.uint8)
    full_img[20:100, 20:100, :] = [50, 100, 200]

    cropped = detect_and_crop_borders(full_img, threshold=10, margin=2)

    assert cropped.ndim == 3
    assert cropped.shape[2] == 3
    assert cropped.shape[0] < 120
    assert cropped.shape[1] < 120
    assert cropped.shape[0] >= 80


def test_crop_completely_dark_image_fallback() -> None:
    """Test that a completely black image does not crash and returns original."""
    dark_img = np.zeros((80, 80), dtype=np.uint8)
    cropped = detect_and_crop_borders(dark_img, threshold=10, margin=2)

    assert cropped.shape == (80, 80)
    assert np.array_equal(cropped, dark_img)


def test_crop_negligible_border_skipped() -> None:
    """Test that if the border is negligible (< 2% area), image is preserved."""
    # Border is only 1 pixel on top
    img = np.ones((100, 100), dtype=np.uint8) * 180
    img[0, :] = 0  # 1% area

    cropped = detect_and_crop_borders(img, threshold=10, margin=2, min_border_ratio=0.02)
    assert cropped.shape == (100, 100)
