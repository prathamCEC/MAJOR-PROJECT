"""
Tests for Metric Normalization.
"""

import pytest

from phase_3_image_quality_assessment.src.config import OCTA_QUALITY_CONFIG, FUNDUS_QUALITY_CONFIG
from phase_3_image_quality_assessment.src.normalization import (
    normalize_blur,
    normalize_brightness,
    normalize_contrast,
    normalize_noise,
    normalize_clipping,
    normalize_content,
    normalize_all_metrics,
)


def test_normalize_blur_bounds() -> None:
    low = normalize_blur(1.0, OCTA_QUALITY_CONFIG)
    high = normalize_blur(800.0, OCTA_QUALITY_CONFIG)

    assert 0.0 <= low <= 100.0
    assert 0.0 <= high <= 100.0
    assert high > low


def test_normalize_brightness_optimal_vs_extreme() -> None:
    optimal = normalize_brightness(100.0, OCTA_QUALITY_CONFIG)
    too_dark = normalize_brightness(5.0, OCTA_QUALITY_CONFIG)

    assert optimal == 100.0
    assert too_dark == 0.0


def test_normalize_noise_clean_vs_noisy() -> None:
    clean = normalize_noise(2.0, OCTA_QUALITY_CONFIG)
    noisy = normalize_noise(30.0, OCTA_QUALITY_CONFIG)

    assert clean == 100.0
    assert noisy == 0.0


def test_normalize_all_metrics_structure() -> None:
    raw = {
        "laplacian_variance": 200.0,
        "mean_brightness": 110.0,
        "rms_contrast": 50.0,
        "noise_residual_std": 3.0,
        "total_clipping_ratio": 0.01,
        "shannon_entropy": 5.5,
        "color_metrics": {"is_color_valid": 1.0, "mean_saturation": 30.0, "color_cast_ratio": 0.5},
    }
    scores = normalize_all_metrics(raw, FUNDUS_QUALITY_CONFIG)

    assert all(0.0 <= s <= 100.0 for s in scores.values())
    assert "blur_score" in scores
    assert "brightness_score" in scores
    assert "color_score" in scores
