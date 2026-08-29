"""
Overall Composite Quality Score Module for Retinal Images.

Computes the weighted composite technical quality index across normalized sub-metrics.
"""

from typing import Dict
import numpy as np

from .config import ModalityQualityConfig


def calculate_overall_quality_score(
    scores: Dict[str, float],
    config: ModalityQualityConfig,
) -> float:
    """
    Compute the composite technical quality score (0 - 100) using modality-specific weights.

    Args:
        scores: Dictionary of normalized scores (0-100) for each dimension.
        config: ModalityQualityConfig with calibrated metric weights.

    Returns:
        Float composite score in [0.0, 100.0].
    """
    w = config.weights
    w.validate()

    blur_s = scores.get("blur_score", 0.0)
    bright_s = scores.get("brightness_score", 0.0)
    contrast_s = scores.get("contrast_score", 0.0)
    noise_s = scores.get("noise_score", 0.0)
    clipping_s = scores.get("clipping_score", 0.0)
    content_s = scores.get("content_score", 0.0)
    color_s = scores.get("color_score", 100.0)

    overall = (
        (w.blur_weight * blur_s)
        + (w.brightness_weight * bright_s)
        + (w.contrast_weight * contrast_s)
        + (w.noise_weight * noise_s)
        + (w.clipping_weight * clipping_s)
        + (w.content_weight * content_s)
        + (w.color_weight * color_s)
    )

    return float(np.clip(overall, 0.0, 100.0))
