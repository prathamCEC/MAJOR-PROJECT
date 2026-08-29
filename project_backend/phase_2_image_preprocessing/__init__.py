"""
Phase 2 — Retinal Image Preprocessing Package

This package provides standardized, modality-aware preprocessing pipelines
for retinal imaging modalities:
- OCT-A (Optical Coherence Tomography Angiography)
- OCT-B (Structural Cross-Sectional OCT)
- Fundus (Color Retinal Fundus Photography)

Prepared for downstream Image Quality Assessment (Phase 3) and
Swin Transformer architectures (Phase 4).
"""

from .src.pipeline import PreprocessPipeline, preprocess_image
from .src.config import (
    ModalityConfig,
    OCTA_CONFIG,
    OCTB_CONFIG,
    FUNDUS_CONFIG,
    SUPPORTED_MODALITIES,
)

__version__ = "1.0.0"

__all__ = [
    "PreprocessPipeline",
    "preprocess_image",
    "ModalityConfig",
    "OCTA_CONFIG",
    "OCTB_CONFIG",
    "FUNDUS_CONFIG",
    "SUPPORTED_MODALITIES",
]
