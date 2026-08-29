"""
Tests for research risk categories, confidence tiers, and uncertainty warnings.
"""

import pytest

from phase_11_report_generator.config import ReportConfig
from phase_11_report_generator.risk_calculator import (
    calculate_risk_category,
    calculate_confidence_category,
    get_uncertainty_statement,
)


def test_risk_category_thresholds():
    cfg = ReportConfig(low_risk_threshold=0.30, moderate_risk_threshold=0.70)

    assert calculate_risk_category(0.15, cfg) == "LOW RISK"
    assert calculate_risk_category(0.50, cfg) == "MODERATE RISK"
    assert calculate_risk_category(0.85, cfg) == "HIGH RISK"


def test_confidence_category_thresholds():
    cfg = ReportConfig(high_confidence_threshold=85.0, moderate_confidence_threshold=65.0)

    assert calculate_confidence_category(90.0, cfg) == "HIGH CONFIDENCE"
    assert calculate_confidence_category(75.0, cfg) == "MODERATE CONFIDENCE"
    assert calculate_confidence_category(50.0, cfg) == "LOW CONFIDENCE"


def test_uncertainty_warning_statement():
    elevated_stmt = get_uncertainty_statement("ELEVATED", is_elevated=True)
    assert "elevated" in elevated_stmt.lower()

    low_stmt = get_uncertainty_statement("LOW", is_elevated=False)
    assert "relatively low" in low_stmt.lower()
