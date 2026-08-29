"""
Centralized Configuration for Phase 3 Retinal Image Quality Assessment.

Defines modality-specific weights, normalization reference ranges, decision
thresholds, warning policies, and dataset path resolution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, Tuple, Union

SUPPORTED_MODALITIES: Set[str] = {"octa", "octb", "fundus"}

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".ppm",
)


@dataclass(frozen=True)
class MetricWeights:
    """
    Weights assigned to individual technical quality metrics.
    Sum of active weights must equal 1.0.
    """
    blur_weight: float = 0.20
    brightness_weight: float = 0.15
    contrast_weight: float = 0.20
    noise_weight: float = 0.15
    clipping_weight: float = 0.10
    content_weight: float = 0.15
    color_weight: float = 0.05

    def validate(self) -> None:
        total = (
            self.blur_weight
            + self.brightness_weight
            + self.contrast_weight
            + self.noise_weight
            + self.clipping_weight
            + self.content_weight
            + self.color_weight
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Metric weights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class ModalityQualityConfig:
    """
    Modality-specific technical quality assessment configuration.
    """
    modality: str
    is_color: bool
    weights: MetricWeights
    accept_threshold: float = 65.0
    warning_threshold: float = 50.0
    warning_policy: str = "approve"  # "approve" (move to approved with warning flag) or "reject"
    
    # Sharpness / Blur reference calibration (Laplacian variance)
    blur_raw_min: float = 10.0
    blur_raw_max: float = 500.0

    # Illumination / Brightness reference range (mean intensity 0-255)
    brightness_opt_low: float = 40.0
    brightness_opt_high: float = 190.0
    brightness_crit_low: float = 15.0
    brightness_crit_high: float = 240.0

    # Contrast reference range (RMS std dev 0-128)
    contrast_opt_low: float = 25.0
    contrast_opt_high: float = 85.0
    contrast_crit_low: float = 8.0

    # Noise residual tolerance (std dev of high-frequency residual)
    noise_clean_max: float = 4.0
    noise_severe_thresh: float = 25.0

    # Clipping fraction tolerance (fraction of pixels <= 2 or >= 253)
    clipping_clean_max: float = 0.02
    clipping_severe_thresh: float = 0.25

    # Content / Shannon Information Entropy reference (bits)
    entropy_min_ref: float = 1.0
    entropy_opt_ref: float = 5.0

    # Color Cast / Saturation limits for Fundus
    color_min_saturation: float = 15.0
    color_max_cast_ratio: float = 0.60


# OCT-A Configuration: Emphasizes sharpness of vascular details and vessel contrast
OCTA_QUALITY_CONFIG = ModalityQualityConfig(
    modality="octa",
    is_color=False,
    weights=MetricWeights(
        blur_weight=0.25,
        brightness_weight=0.15,
        contrast_weight=0.20,
        noise_weight=0.15,
        clipping_weight=0.10,
        content_weight=0.15,
        color_weight=0.0,
    ),
    accept_threshold=65.0,
    warning_threshold=50.0,
    warning_policy="approve",
    blur_raw_min=15.0,
    blur_raw_max=600.0,
)

# OCT-B Configuration: Emphasizes structural layer contrast and layer boundary definition
OCTB_QUALITY_CONFIG = ModalityQualityConfig(
    modality="octb",
    is_color=False,
    weights=MetricWeights(
        blur_weight=0.20,
        brightness_weight=0.15,
        contrast_weight=0.25,
        noise_weight=0.15,
        clipping_weight=0.10,
        content_weight=0.15,
        color_weight=0.0,
    ),
    accept_threshold=65.0,
    warning_threshold=50.0,
    warning_policy="approve",
    blur_raw_min=10.0,
    blur_raw_max=450.0,
)

# Fundus Configuration: Includes color balance, macula/optic disc illumination, and sharpness
FUNDUS_QUALITY_CONFIG = ModalityQualityConfig(
    modality="fundus",
    is_color=True,
    weights=MetricWeights(
        blur_weight=0.20,
        brightness_weight=0.15,
        contrast_weight=0.15,
        noise_weight=0.10,
        clipping_weight=0.10,
        content_weight=0.15,
        color_weight=0.15,
    ),
    accept_threshold=65.0,
    warning_threshold=50.0,
    warning_policy="approve",
    blur_raw_min=15.0,
    blur_raw_max=500.0,
)

MODALITY_QUALITY_CONFIG_MAP: Dict[str, ModalityQualityConfig] = {
    "octa": OCTA_QUALITY_CONFIG,
    "octb": OCTB_QUALITY_CONFIG,
    "fundus": FUNDUS_QUALITY_CONFIG,
}


def get_modality_quality_config(modality: str) -> ModalityQualityConfig:
    """
    Retrieve the Phase 3 quality configuration for a given modality.

    Args:
        modality: Modality identifier ('octa', 'octb', 'fundus').

    Returns:
        ModalityQualityConfig object.

    Raises:
        ValueError: If modality is unsupported.
    """
    clean_modality = str(modality).strip().lower()
    if clean_modality not in MODALITY_QUALITY_CONFIG_MAP:
        raise ValueError(
            f"Unsupported modality: '{modality}'. "
            f"Expected one of: {SUPPORTED_MODALITIES}"
        )
    return MODALITY_QUALITY_CONFIG_MAP[clean_modality]


def get_project_backend_root() -> Path:
    """Get the absolute Path of the project_backend directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_default_processed_input_dir(modality: str | None = None) -> Path:
    """Get the default processed dataset input directory."""
    base = get_project_backend_root() / "datasets" / "processed"
    if modality:
        return base / modality.lower()
    return base


def get_default_approved_dir(modality: str | None = None) -> Path:
    """Get the default approved dataset directory."""
    base = get_project_backend_root() / "datasets" / "approved"
    if modality:
        return base / modality.lower()
    return base


def get_default_rejected_dir(modality: str | None = None) -> Path:
    """Get the default rejected dataset directory."""
    base = get_project_backend_root() / "datasets" / "rejected"
    if modality:
        return base / modality.lower()
    return base


def get_default_phase3_log_dir() -> Path:
    """Get the default logs directory for Phase 3 output CSV/JSON."""
    log_dir = get_project_backend_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
