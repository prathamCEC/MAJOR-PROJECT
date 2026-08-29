"""
Validation Module for Phase 2 Retinal Image Preprocessing.

Provides strict pre- and post-processing validation, data type verification,
finite value checks, dimensional integrity checks, and custom domain exceptions.
"""

from typing import Tuple, Optional
import numpy as np

from .config import SUPPORTED_MODALITIES, TARGET_WIDTH, TARGET_HEIGHT


class PreprocessingError(Exception):
    """Base exception for all Phase 2 preprocessing failures."""
    pass


class ImageValidationError(PreprocessingError):
    """Raised when an input or intermediate image fails validation checks."""
    pass


class CorruptedImageError(PreprocessingError):
    """Raised when an image file cannot be read, decoded, or is corrupted."""
    pass


class InvalidModalityError(PreprocessingError):
    """Raised when an unsupported imaging modality is specified."""
    pass


class BorderCropError(PreprocessingError):
    """Raised when border detection or cropping encounters an invalid state."""
    pass


class ImageSaveError(PreprocessingError):
    """Raised when a processed image cannot be written or verified on disk."""
    pass


def validate_modality(modality: str) -> str:
    """
    Validate that the specified modality is supported.

    Args:
        modality: String modality identifier.

    Returns:
        Cleaned lowercase modality string.

    Raises:
        InvalidModalityError: If modality is not in SUPPORTED_MODALITIES.
    """
    if not isinstance(modality, str):
        raise InvalidModalityError(
            f"Modality must be a string, got {type(modality).__name__}"
        )
    clean_modality = modality.strip().lower()
    if clean_modality not in SUPPORTED_MODALITIES:
        raise InvalidModalityError(
            f"Invalid modality '{modality}'. Supported modalities are: {SUPPORTED_MODALITIES}"
        )
    return clean_modality


def validate_raw_image(image: Optional[np.ndarray], modality: str) -> None:
    """
    Validate an image immediately after loading from disk.

    Args:
        image: Numpy ndarray representing the loaded image.
        modality: Target modality identifier ("octa", "octb", "fundus").

    Raises:
        ImageValidationError: If image is None, empty, invalid dimensions,
                              contains non-finite values (NaN/Inf), or unexpected channels.
    """
    if image is None:
        raise ImageValidationError("Image array is None.")

    if not isinstance(image, np.ndarray):
        raise ImageValidationError(
            f"Image must be a numpy.ndarray, got {type(image).__name__}"
        )

    if image.size == 0:
        raise ImageValidationError("Image array is empty (size == 0).")

    if image.ndim < 2 or image.ndim > 3:
        raise ImageValidationError(
            f"Image must have 2 or 3 dimensions, got ndim={image.ndim} with shape {image.shape}."
        )

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageValidationError(
            f"Invalid image dimensions: height={height}, width={width}."
        )

    # Check for NaN and Inf values
    if not np.isfinite(image).all():
        has_nan = np.isnan(image).any()
        has_inf = np.isinf(image).any()
        raise ImageValidationError(
            f"Image contains non-finite values (has_nan={has_nan}, has_inf={has_inf})."
        )

    # Check modality channel requirements
    modality_clean = validate_modality(modality)
    if modality_clean == "fundus":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ImageValidationError(
                f"Fundus image must have 3 color channels, got shape {image.shape}."
            )
    else:  # octa or octb
        if image.ndim == 3 and image.shape[2] not in (1, 3):
            raise ImageValidationError(
                f"{modality_clean.upper()} image must be 1-channel or 3-channel, got shape {image.shape}."
            )

    # Check supported dtypes
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ImageValidationError(
            f"Unsupported image dtype: {image.dtype}. Supported: uint8, uint16, float32, float64."
        )


def validate_processed_image(
    image: Optional[np.ndarray],
    target_size: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
    expected_channels: int = 3,
) -> None:
    """
    Validate a preprocessed image before saving and for Swin Transformer readiness.

    Args:
        image: Numpy ndarray of the processed image.
        target_size: Expected (width, height) dimensions.
        expected_channels: Expected channel count (default 3 for Swin-T compatibility).

    Raises:
        ImageValidationError: If dimensions, data type, value range, or finiteness fail.
    """
    if image is None:
        raise ImageValidationError("Processed image is None.")

    if not isinstance(image, np.ndarray):
        raise ImageValidationError(
            f"Processed image must be a numpy.ndarray, got {type(image).__name__}"
        )

    if not np.isfinite(image).all():
        raise ImageValidationError("Processed image contains NaN or Inf values.")

    target_w, target_h = target_size
    if image.ndim != 3:
        raise ImageValidationError(
            f"Processed image must have 3 dimensions (H, W, C), got ndim={image.ndim} shape {image.shape}."
        )

    height, width, channels = image.shape
    if height != target_h or width != target_w:
        raise ImageValidationError(
            f"Processed image dimensions mismatch: expected ({target_h}, {target_w}), got ({height}, {width})."
        )

    if channels != expected_channels:
        raise ImageValidationError(
            f"Processed image channels mismatch: expected {expected_channels}, got {channels}."
        )

    if image.dtype != np.uint8:
        raise ImageValidationError(
            f"Processed image dtype must be uint8, got {image.dtype}."
        )

    # Verify pixel intensity range is strictly within [0, 255]
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if min_val < 0 or max_val > 255:
        raise ImageValidationError(
            f"Processed image pixel values out of uint8 bounds: min={min_val}, max={max_val}."
        )
