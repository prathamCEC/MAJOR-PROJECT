"""
Confidence Estimation Module for Phase 9.

Transforms Monte Carlo prediction variance and dispersion into a bounded, normalized
model confidence score for research and exploratory analysis.
"""

from typing import Dict, Tuple
import torch


def calculate_confidence(
    variance: torch.Tensor,
    uncertainty_scale: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """
    Calculate normalized model confidence score from predictive variance.

    Formula:
        normalized_uncertainty = clamp(variance / uncertainty_scale, 0.0, 1.0)
        confidence = 1.0 - normalized_uncertainty
        confidence_percent = confidence * 100.0

    Note:
        The maximum theoretical variance for a Bernoulli binary variable occurs at p=0.5
        with variance = p*(1-p) = 0.25. Setting uncertainty_scale=0.25 standardizes
        dispersion across the entire theoretical domain.

    RESEARCH DISCLAIMER:
        This confidence score is a model uncertainty-derived research metric and is
        NOT a clinically calibrated probability of correctness.

    Args:
        variance: Predictive variance tensor [B]
        uncertainty_scale: Maximum scaling factor (default: 0.25)

    Returns:
        Dict containing:
        - 'confidence': Tensor [B] in range [0.0, 1.0]
        - 'confidence_percent': Tensor [B] in range [0.0, 100.0]
    """
    if uncertainty_scale <= 0.0:
        raise ValueError(f"uncertainty_scale must be > 0, got {uncertainty_scale}.")

    # Normalize variance against theoretical dispersion scale
    normalized_uncertainty = torch.clamp(variance / uncertainty_scale, min=0.0, max=1.0)

    # Invert to obtain confidence (lower uncertainty -> higher confidence)
    confidence = 1.0 - normalized_uncertainty
    confidence_percent = confidence * 100.0

    return {
        "confidence": confidence,
        "confidence_percent": confidence_percent,
    }
