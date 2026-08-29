"""
Tests for Fundus Color Quality Assessment.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.color_quality import compute_color_quality_metrics


def test_valid_color_fundus() -> None:
    # Synthetic fundus with natural color spread
    fundus = np.zeros((80, 80, 3), dtype=np.uint8)
    fundus[:, :, 2] = 190  # R
    fundus[:, :, 1] = 90   # G
    fundus[:, :, 0] = 30   # B

    res = compute_color_quality_metrics(fundus, is_color=True)
    assert res["is_color_valid"] == 1.0
    assert res["mean_saturation"] > 10.0


def test_fake_grayscale_fundus_detected() -> None:
    # 3 identical channels (monochrome)
    mono = np.zeros((80, 80, 3), dtype=np.uint8)
    mono[:, :, 0] = 120
    mono[:, :, 1] = 120
    mono[:, :, 2] = 120

    res = compute_color_quality_metrics(mono, is_color=True)
    assert res["is_color_valid"] == 0.0
