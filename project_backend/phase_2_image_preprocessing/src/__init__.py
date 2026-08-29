"""
Phase 2 Preprocessing Core Source Modules
"""

from .config import (
    ModalityConfig,
    OCTA_CONFIG,
    OCTB_CONFIG,
    FUNDUS_CONFIG,
    SUPPORTED_MODALITIES,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)
from .validation import (
    PreprocessingError,
    ImageValidationError,
    CorruptedImageError,
    InvalidModalityError,
    BorderCropError,
    ImageSaveError,
    validate_modality,
    validate_raw_image,
    validate_processed_image,
)
from .image_loader import load_image
from .border_crop import detect_and_crop_borders
from .clahe import apply_clahe
from .gaussian_filter import apply_gaussian_filter
from .median_filter import apply_median_filter
from .normalization import normalize_to_float32, convert_to_uint8
from .resizing import resize_with_aspect_ratio_and_pad
from .utils import (
    find_image_files,
    get_processed_filename,
    ensure_three_channels,
    verify_saved_image,
)
from .pipeline import PreprocessPipeline, preprocess_image

__all__ = [
    "ModalityConfig",
    "OCTA_CONFIG",
    "OCTB_CONFIG",
    "FUNDUS_CONFIG",
    "SUPPORTED_MODALITIES",
    "TARGET_WIDTH",
    "TARGET_HEIGHT",
    "PreprocessingError",
    "ImageValidationError",
    "CorruptedImageError",
    "InvalidModalityError",
    "BorderCropError",
    "ImageSaveError",
    "validate_modality",
    "validate_raw_image",
    "validate_processed_image",
    "load_image",
    "detect_and_crop_borders",
    "apply_clahe",
    "apply_gaussian_filter",
    "apply_median_filter",
    "normalize_to_float32",
    "convert_to_uint8",
    "resize_with_aspect_ratio_and_pad",
    "find_image_files",
    "get_processed_filename",
    "ensure_three_channels",
    "verify_saved_image",
    "PreprocessPipeline",
    "preprocess_image",
    "BatchProcessor",
]
