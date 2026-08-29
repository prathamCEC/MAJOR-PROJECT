"""
Tests for Decision Engine.
"""

import pytest

from phase_3_image_quality_assessment.src.config import OCTA_QUALITY_CONFIG
from phase_3_image_quality_assessment.src.decision import make_decision, DecisionEnum


def test_decision_accept() -> None:
    scores = {
        "blur_score": 85.0,
        "brightness_score": 90.0,
        "contrast_score": 80.0,
        "noise_score": 90.0,
        "clipping_score": 95.0,
        "content_score": 85.0,
        "color_score": 100.0,
    }
    decision = make_decision(85.0, scores, OCTA_QUALITY_CONFIG)
    assert decision.decision == DecisionEnum.ACCEPT
    assert decision.is_approved_for_ai is True


def test_decision_warning() -> None:
    scores = {
        "blur_score": 55.0,
        "brightness_score": 60.0,
        "contrast_score": 55.0,
        "noise_score": 60.0,
        "clipping_score": 65.0,
        "content_score": 55.0,
        "color_score": 100.0,
    }
    decision = make_decision(58.0, scores, OCTA_QUALITY_CONFIG)
    assert decision.decision == DecisionEnum.WARNING


def test_decision_reject_low_score() -> None:
    scores = {
        "blur_score": 35.0,
        "brightness_score": 40.0,
        "contrast_score": 30.0,
        "noise_score": 40.0,
        "clipping_score": 45.0,
        "content_score": 35.0,
        "color_score": 100.0,
    }
    decision = make_decision(35.0, scores, OCTA_QUALITY_CONFIG)
    assert decision.decision == DecisionEnum.REJECT
    assert decision.is_approved_for_ai is False


def test_decision_hard_failure_overrides_high_score() -> None:
    # High overall average, but zero content/entropy
    scores = {
        "blur_score": 95.0,
        "brightness_score": 95.0,
        "contrast_score": 90.0,
        "noise_score": 95.0,
        "clipping_score": 95.0,
        "content_score": 5.0,  # Critical failure
        "color_score": 100.0,
    }
    decision = make_decision(80.0, scores, OCTA_QUALITY_CONFIG)
    assert decision.decision == DecisionEnum.REJECT
    assert decision.is_approved_for_ai is False
    assert any("entropy" in f for f in decision.failed_checks)
