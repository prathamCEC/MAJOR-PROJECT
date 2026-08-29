"""
Tests for Clipping and Saturation Detection.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.clipping import compute_clipping_metrics


def test_clipping_detection() -> None:
    # 50% saturated to 255, 50% normal
    clipped = np.ones((100, 100), dtype=np.uint8) * 128
    clipped[:50, :] = 255

    res = compute_clipping_metrics(clipped, is_color=False)
    assert pytest.approx(res["overexposed_clipping_ratio"], 0.05) == 0.50
    assert pytest.approx(res["total_clipping_ratio"], 0.05) == 0.50


def test_unclipped_image() -> None:
    unclipped = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
    res = compute_clipping_metrics(unclipped, is_color=False)
    assert res["total_clipping_ratio"] == 0.0
