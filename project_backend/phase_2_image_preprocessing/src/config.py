"""
Centralized Configuration for Phase 2 Retinal Image Preprocessing.

Defines modality-specific dataclasses, global defaults, supported extensions,
and standardized target dimensions for OCT-A, OCT-B, and Fundus modalities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union
import cv2

# Global target dimensions for downstream Swin Transformer models
TARGET_WIDTH: int = 224
TARGET_HEIGHT: int = 224

SUPPORTED_MODALITIES: Tuple[str, ...] = ("octa", "octb", "fundus")

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
class ModalityConfig:
    """
    Modality-specific configuration parameters.

    Attributes:
        modality: Name identifier ("octa", "octb", "fundus").
        is_color: Whether the modality contains native color information.
        target_size: (width, height) for final output dimensions.
        apply_border_crop: Whether to perform black border detection & cropping.
        crop_threshold: Intensity threshold below which pixels are considered border.
        crop_margin: Inward safety margin (pixels) to prevent cutting clinical details.
        min_border_ratio: Minimum ratio of border pixels to trigger cropping.
        apply_clahe: Whether to apply CLAHE contrast enhancement.
        clahe_clip_limit: Threshold for contrast limiting in CLAHE.
        clahe_tile_grid_size: Grid size for histogram equalization (rows, cols).
        apply_gaussian: Whether to apply conservative Gaussian smoothing.
        gaussian_kernel_size: Gaussian filter kernel dimensions (must be odd).
        gaussian_sigma: Gaussian kernel standard deviation.
        apply_median: Whether to apply conservative Median filtering.
        median_kernel_size: Median filter aperture linear size (must be odd).
        norm_min: Lower bound for float32 normalization.
        norm_max: Upper bound for float32 normalization.
        pad_value: Fill value for letterbox padding (0 or tuple of 0s).
        resize_interpolation_down: OpenCV interpolation method for downscaling.
        resize_interpolation_up: OpenCV interpolation method for upscaling.
        convert_to_3_channel: Replicate grayscale to 3 channels for Swin Transformer.
        output_format: Default output file format ("png").
        overwrite_existing: Whether to overwrite existing processed files.
    """

    modality: str
    is_color: bool
    target_size: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)
    apply_border_crop: bool = True
    crop_threshold: int = 10
    crop_margin: int = 2
    min_border_ratio: float = 0.02
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_gaussian: bool = True
    gaussian_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.0
    apply_median: bool = False
    median_kernel_size: int = 3
    norm_min: float = 0.0
    norm_max: float = 1.0
    pad_value: Union[int, Tuple[int, int, int]] = 0
    resize_interpolation_down: int = cv2.INTER_AREA
    resize_interpolation_up: int = cv2.INTER_CUBIC
    convert_to_3_channel: bool = True
    output_format: str = "png"
    overwrite_existing: bool = False


# OCT-A Configuration: Focus on microvasculature & capillary network preservation
OCTA_CONFIG = ModalityConfig(
    modality="octa",
    is_color=False,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
    apply_border_crop=True,
    crop_threshold=10,
    crop_margin=2,
    apply_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    apply_gaussian=True,
    gaussian_kernel_size=(3, 3),
    gaussian_sigma=0.0,
    apply_median=False,  # Avoid median blur to protect delicate capillary lines
    median_kernel_size=3,
    pad_value=0,
    convert_to_3_channel=True,
)

# OCT-B Configuration: Focus on cross-sectional retinal layer boundary preservation
OCTB_CONFIG = ModalityConfig(
    modality="octb",
    is_color=False,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
    apply_border_crop=True,
    crop_threshold=10,
    crop_margin=2,
    apply_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    apply_gaussian=True,
    gaussian_kernel_size=(3, 3),
    gaussian_sigma=0.0,
    apply_median=True,  # Conservative median blur reduces speckle noise along layers
    median_kernel_size=3,
    pad_value=0,
    convert_to_3_channel=True,
)

# Fundus Configuration: Color preservation (LAB L-channel CLAHE), optic disc & macula preservation
FUNDUS_CONFIG = ModalityConfig(
    modality="fundus",
    is_color=True,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
    apply_border_crop=True,
    crop_threshold=10,
    crop_margin=2,
    apply_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    apply_gaussian=True,
    gaussian_kernel_size=(3, 3),
    gaussian_sigma=0.0,
    apply_median=False,  # Color fundus details preserved without median blurring
    median_kernel_size=3,
    pad_value=(0, 0, 0),
    convert_to_3_channel=True,
)

MODALITY_CONFIG_MAP = {
    "octa": OCTA_CONFIG,
    "octb": OCTB_CONFIG,
    "fundus": FUNDUS_CONFIG,
}


def get_modality_config(modality: str) -> ModalityConfig:
    """
    Retrieve the configuration for a given modality.

    Args:
        modality: Modality identifier ('octa', 'octb', 'fundus').

    Returns:
        ModalityConfig object.

    Raises:
        ValueError: If modality is unsupported.
    """
    clean_modality = str(modality).strip().lower()
    if clean_modality not in MODALITY_CONFIG_MAP:
        raise ValueError(
            f"Unsupported modality: '{modality}'. "
            f"Expected one of: {SUPPORTED_MODALITIES}"
        )
    return MODALITY_CONFIG_MAP[clean_modality]


def get_project_backend_root() -> Path:
    """
    Get the absolute Path of the project_backend directory.
    Uses pathlib for cross-platform (Windows/Linux/macOS) compatibility.
    """
    # config.py is at: project_backend/phase_2_image_preprocessing/src/config.py
    # Root project_backend is 3 levels up
    return Path(__file__).resolve().parent.parent.parent


def get_default_raw_dir(modality: str | None = None) -> Path:
    """Get the default raw dataset directory."""
    base = get_project_backend_root() / "datasets" / "raw"
    if modality:
        return base / modality.lower()
    return base


def get_default_processed_dir(modality: str | None = None) -> Path:
    """Get the default processed dataset directory."""
    base = get_project_backend_root() / "datasets" / "processed"
    if modality:
        return base / modality.lower()
    return base


def get_default_log_file() -> Path:
    """Get the default log file path for failed images."""
    log_dir = get_project_backend_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "phase2_failed_images.txt"
