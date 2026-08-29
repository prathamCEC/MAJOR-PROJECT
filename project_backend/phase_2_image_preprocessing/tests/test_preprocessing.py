"""
Tests for Core Preprocessing Filters (CLAHE, Gaussian, Median).
"""

import cv2
import numpy as np
import pytest

from phase_2_image_preprocessing.src.clahe import apply_clahe
from phase_2_image_preprocessing.src.gaussian_filter import apply_gaussian_filter
from phase_2_image_preprocessing.src.median_filter import apply_median_filter


def test_clahe_grayscale() -> None:
    """Test CLAHE on grayscale OCT image."""
    img = np.linspace(50, 150, num=10000, dtype=np.uint8).reshape((100, 100))
    enhanced = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8), is_color=False)

    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8
    # Contrast/dynamic range should be expanded or redistributed
    assert (int(np.max(enhanced)) - int(np.min(enhanced))) >= (int(np.max(img)) - int(np.min(img)))


def test_clahe_color_fundus_preserves_color() -> None:
    """Test that CLAHE on Fundus operates in LAB color space and preserves color channels."""
    # Synthetic fundus-like image (predominantly red/orange in RGB)
    fundus_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    fundus_bgr[:, :, 0] = 30   # Blue
    fundus_bgr[:, :, 1] = 80   # Green
    fundus_bgr[:, :, 2] = 200  # Red

    enhanced = apply_clahe(fundus_bgr, clip_limit=2.0, tile_grid_size=(8, 8), is_color=True)

    assert enhanced.shape == (100, 100, 3)
    assert enhanced.dtype == np.uint8
    # Red channel should still dominate over blue channel
    assert np.mean(enhanced[:, :, 2]) > np.mean(enhanced[:, :, 0])


def test_gaussian_filter() -> None:
    """Test Gaussian filter smoothing behavior."""
    # Create image with high-frequency noise spike
    img = np.ones((50, 50), dtype=np.uint8) * 100
    img[25, 25] = 250

    filtered = apply_gaussian_filter(img, kernel_size=(3, 3), sigma=0.0)

    assert filtered.shape == (50, 50)
    assert filtered.dtype == np.uint8
    # The spike should be smoothed
    assert filtered[25, 25] < 250
    # Surrounding pixels should remain close to 100
    assert filtered[0, 0] == 100


def test_median_filter() -> None:
    """Test Median filter removal of salt-and-pepper noise."""
    img = np.ones((50, 50), dtype=np.uint8) * 128
    # Add salt and pepper isolated noise
    img[10, 10] = 255
    img[20, 20] = 0

    filtered = apply_median_filter(img, kernel_size=3)

    assert filtered.shape == (50, 50)
    # Isolated noise pixels should be completely removed by median filter
    assert filtered[10, 10] == 128
    assert filtered[20, 20] == 128
