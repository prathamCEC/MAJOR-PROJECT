"""
Tests for Quality Score Computation.
"""

import pytest

from phase_3_image_quality_assessment.src.config import OCTA_QUALITY_CONFIG
from phase_3_image_quality_assessment.src.quality_score import calculate_overall_quality_score


def test_perfect_scores_yield_100() -> None:
    perfect_scores = {
        "blur_score": 100.0,
        "brightness_score": 100.0,
        "contrast_score": 100.0,
        "noise_score": 100.0,
        "clipping_score": 100.0,
        "content_score": 100.0,
        "color_score": 100.0,
    }
    overall = calculate_overall_quality_score(perfect_scores, OCTA_QUALITY_CONFIG)
    assert pytest.approx(overall, 0.01) == 100.0


def test_zero_scores_yield_zero() -> None:
    zero_scores = {
        "blur_score": 0.0,
        "brightness_score": 0.0,
        "contrast_score": 0.0,
        "noise_score": 0.0,
        "clipping_score": 0.0,
        "content_score": 0.0,
        "color_score": 0.0,
    }
    overall = calculate_overall_quality_score(zero_scores, OCTA_QUALITY_CONFIG)
    assert pytest.approx(overall, 0.01) == 0.0
