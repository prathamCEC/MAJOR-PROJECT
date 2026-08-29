"""
Risk Categorization and Confidence Mapping Engine for Research Reports.

Provides configurable research decision-support classification logic based on
predicted disease probabilities and Phase 9 Monte Carlo Dropout uncertainty statistics.
"""

from typing import Tuple
from .config import ReportConfig, get_default_report_config


def calculate_risk_category(probability: float, config: ReportConfig = None) -> str:
    """
    Calculate research risk category from predicted disease probability.

    Args:
        probability: Model predicted probability in [0.0, 1.0].
        config: ReportConfig containing low and moderate risk cutoffs.

    Returns:
        Risk category string: 'LOW RISK', 'MODERATE RISK', or 'HIGH RISK'.
    """
    cfg = config or get_default_report_config()

    if probability < cfg.low_risk_threshold:
        return "LOW RISK"
    elif probability < cfg.moderate_risk_threshold:
        return "MODERATE RISK"
    else:
        return "HIGH RISK"


def calculate_confidence_category(confidence_percent: float, config: ReportConfig = None) -> str:
    """
    Calculate model confidence tier from Phase 9 confidence score.

    Args:
        confidence_percent: Confidence in [0.0, 100.0]%.
        config: ReportConfig containing high and moderate thresholds.

    Returns:
        Confidence tier string: 'HIGH CONFIDENCE', 'MODERATE CONFIDENCE', or 'LOW CONFIDENCE'.
    """
    cfg = config or get_default_report_config()

    if confidence_percent >= cfg.high_confidence_threshold:
        return "HIGH CONFIDENCE"
    elif confidence_percent >= cfg.moderate_confidence_threshold:
        return "MODERATE CONFIDENCE"
    else:
        return "LOW CONFIDENCE"


def get_uncertainty_statement(uncertainty_level: str, is_elevated: bool) -> str:
    """
    Generate an objective, research-compliant statement regarding predictive uncertainty.

    Args:
        uncertainty_level: Level string ('LOW', 'MODERATE', 'ELEVATED').
        is_elevated: Boolean indicating whether variance exceeded alerting thresholds.

    Returns:
        Objective description string.
    """
    if is_elevated or uncertainty_level.upper() == "ELEVATED":
        return "Prediction uncertainty is elevated across stochastic Monte Carlo dropout passes."
    elif uncertainty_level.upper() == "MODERATE":
        return "Model uncertainty is moderate across stochastic forward passes."
    else:
        return "Model uncertainty is relatively low across stochastic forward passes."
