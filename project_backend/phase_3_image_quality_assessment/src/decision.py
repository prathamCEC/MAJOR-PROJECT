"""
Quality Decision Engine for Phase 3 Retinal Image Quality Assessment.

Evaluates composite and dimensional quality scores against calibrated thresholds
to issue ACCEPT / WARNING / REJECT decisions and approval routing status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from .config import ModalityQualityConfig


class DecisionEnum(str, Enum):
    """Quality decision outcomes."""
    ACCEPT = "ACCEPT"
    WARNING = "WARNING"
    REJECT = "REJECT"


@dataclass(frozen=True)
class QualityDecision:
    """
    Structured quality assessment decision.

    Attributes:
        decision: DecisionEnum (ACCEPT, WARNING, or REJECT).
        overall_score: Composite quality score (0.0 - 100.0).
        is_approved_for_ai: Whether image is routed to approved dataset for Phase 4.
        reason: Human-readable rationale for the decision.
        failed_checks: List of specific sub-metrics that triggered penalties.
    """
    decision: DecisionEnum
    overall_score: float
    is_approved_for_ai: bool
    reason: str
    failed_checks: List[str] = field(default_factory=list)


def make_decision(
    overall_score: float,
    scores: Dict[str, float],
    config: ModalityQualityConfig,
) -> QualityDecision:
    """
    Evaluate overall and sub-metric scores to produce a standardized quality decision.

    Args:
        overall_score: Composite score in [0.0, 100.0].
        scores: Dictionary of normalized sub-scores.
        config: ModalityQualityConfig object.

    Returns:
        QualityDecision dataclass.
    """
    failed_checks: List[str] = []

    # 1. Check for critical hard failures (severe defects that disqualify image regardless of overall score)
    if scores.get("content_score", 100.0) < 20.0:
        failed_checks.append("Insufficient diagnostic information/entropy")
    if scores.get("brightness_score", 100.0) < 15.0:
        failed_checks.append("Severe over/underexposure")
    if scores.get("blur_score", 100.0) < 15.0:
        failed_checks.append("Severe image defocus/blur")
    if scores.get("clipping_score", 100.0) < 15.0:
        failed_checks.append("Severe saturation/clipping")
    if config.is_color and scores.get("color_score", 100.0) < 25.0:
        failed_checks.append("Degenerate color or monochrome representation in Fundus")

    # If critical hard failures exist, issue immediate REJECT
    if failed_checks:
        reason = f"Rejected due to critical failure(s): {', '.join(failed_checks)}."
        return QualityDecision(
            decision=DecisionEnum.REJECT,
            overall_score=overall_score,
            is_approved_for_ai=False,
            reason=reason,
            failed_checks=failed_checks,
        )

    # 2. Threshold-based decision logic
    if overall_score >= config.accept_threshold:
        decision = DecisionEnum.ACCEPT
        is_approved = True
        reason = "Meets all technical quality criteria for downstream AI processing."
    elif overall_score >= config.warning_threshold:
        decision = DecisionEnum.WARNING
        # Determine approval based on warning policy
        if config.warning_policy == "approve":
            is_approved = True
            reason = (
                f"Borderline technical quality (score: {overall_score:.1f}); "
                "provisionally approved under 'approve' warning policy."
            )
        else:
            is_approved = False
            reason = (
                f"Borderline technical quality (score: {overall_score:.1f}); "
                "rejected under strict 'reject' warning policy."
            )
    else:
        decision = DecisionEnum.REJECT
        is_approved = False
        reason = (
            f"Overall quality score ({overall_score:.1f}) is below warning threshold "
            f"({config.warning_threshold:.1f})."
        )

    return QualityDecision(
        decision=decision,
        overall_score=overall_score,
        is_approved_for_ai=is_approved,
        reason=reason,
        failed_checks=failed_checks,
    )
