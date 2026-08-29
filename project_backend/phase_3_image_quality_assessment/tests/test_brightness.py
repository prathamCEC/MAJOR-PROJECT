"""
Tests for Brightness and Illumination Metric Extraction.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.brightness import compute_brightness_metrics


def test_brightness_dark_vs_bright() -> None:
    dark = np.ones((50, 50), dtype=np.uint8) * 30
    bright = np.ones((50, 50), dtype=np.uint8) * 210

    dark_res = compute_brightness_metrics(dark)
    bright_res = compute_brightness_metrics(bright)

    assert pytest.approx(dark_res["mean_brightness"], 1.0) == 30.0
    assert pytest.approx(bright_res["mean_brightness"], 1.0) == 210.0
    assert dark_res["mean_brightness"] < bright_res["mean_brightness"]


def test_color_fundus_luminance() -> None:
    fundus = np.zeros((50, 50, 3), dtype=np.uint8)
    fundus[:, :, 2] = 180  # R
    fundus[:, :, 1] = 80   # G
    fundus[:, :, 0] = 30   # B

    res = compute_brightness_metrics(fundus, is_color=True)
    assert res["mean_brightness"] > 0.0
