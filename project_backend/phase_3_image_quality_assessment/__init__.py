"""
Phase 3 — Retinal Image Quality Assessment Package

Provides comprehensive, non-destructive image quality evaluation for:
- OCT-A (Vascular Angiography)
- OCT-B (Cross-Sectional Structural OCT)
- Fundus (Color Fundus Photography)

Calculates normalized technical quality metrics (blur, brightness, contrast,
noise, clipping, content, and color), computes composite quality scores, and
issues ACCEPT / WARNING / REJECT decisions for downstream AI suitability.
"""

from .src.pipeline import QualityAssessmentPipeline, assess_image, assess_image_file
from .src.config import (
    ModalityQualityConfig,
    OCTA_QUALITY_CONFIG,
    OCTB_QUALITY_CONFIG,
    FUNDUS_QUALITY_CONFIG,
    SUPPORTED_MODALITIES,
)
from .src.decision import QualityDecision, DecisionEnum

__version__ = "1.0.0"

__all__ = [
    "QualityAssessmentPipeline",
    "assess_image",
    "assess_image_file",
    "ModalityQualityConfig",
    "OCTA_QUALITY_CONFIG",
    "OCTB_QUALITY_CONFIG",
    "FUNDUS_QUALITY_CONFIG",
    "SUPPORTED_MODALITIES",
    "QualityDecision",
    "DecisionEnum",
]
