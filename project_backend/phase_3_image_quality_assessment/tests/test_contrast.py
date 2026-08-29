"""
Tests for Contrast Metric Extraction.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.contrast import compute_contrast_metrics


def test_contrast_high_vs_low() -> None:
    # Low contrast: small variations around 100
    low = np.random.randint(95, 105, size=(60, 60), dtype=np.uint8)
    # High contrast: full dynamic range
    high = np.random.randint(10, 240, size=(60, 60), dtype=np.uint8)

    low_res = compute_contrast_metrics(low)
    high_res = compute_contrast_metrics(high)

    assert high_res["rms_contrast"] > low_res["rms_contrast"]
    assert high_res["dynamic_range"] > low_res["dynamic_range"]
