"""
Validation Module for Phase 3 Retinal Image Quality Assessment.

Ensures that input image arrays meet structural and numerical preconditions
prior to technical metric computation without modifying the input tensors.
"""

from typing import Optional
import numpy as np

from .config import SUPPORTED_MODALITIES


class QualityAssessmentError(Exception):
    """Base exception for Phase 3 quality assessment failures."""
    pass


class ImageValidationError(QualityAssessmentError):
    """Raised when an input image fails structural or numerical validation."""
    pass


class CorruptedImageError(QualityAssessmentError):
    """Raised when an image file cannot be read, decoded, or is damaged."""
    pass


class InvalidModalityError(QualityAssessmentError):
    """Raised when an unsupported imaging modality is specified."""
    pass


def validate_modality(modality: str) -> str:
    """
    Validate that the modality is supported by Phase 3.

    Args:
        modality: String modality identifier.

    Returns:
        Cleaned lowercase modality string.

    Raises:
        InvalidModalityError: If modality is invalid or unsupported.
    """
    if not isinstance(modality, str):
        raise InvalidModalityError(
            f"Modality must be a string, got {type(modality).__name__}"
        )
    clean = modality.strip().lower()
    if clean not in SUPPORTED_MODALITIES:
        raise InvalidModalityError(
            f"Invalid modality '{modality}'. Supported modalities are: {sorted(list(SUPPORTED_MODALITIES))}"
        )
    return clean


def validate_assessment_image(image: Optional[np.ndarray], modality: str) -> None:
    """
    Validate an image prior to running quality assessment metrics.

    Args:
        image: Numpy ndarray of the retinal image.
        modality: Imaging modality identifier ('octa', 'octb', 'fundus').

    Raises:
        ImageValidationError: If image is None, empty, non-finite, degenerate,
                              or has invalid channels.
    """
    if image is None:
        raise ImageValidationError("Assessment image is None.")

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

    if not np.isfinite(image).all():
        has_nan = np.isnan(image).any()
        has_inf = np.isinf(image).any()
        raise ImageValidationError(
            f"Image contains non-finite values (has_nan={has_nan}, has_inf={has_inf})."
        )

    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ImageValidationError(
            f"Unsupported image dtype: {image.dtype}. Supported: uint8, uint16, float32, float64."
        )

    mod_clean = validate_modality(modality)

    # Modality channel checks
    if mod_clean == "fundus":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ImageValidationError(
                f"Fundus image must have 3 color channels, got shape {image.shape}."
            )
    else:  # octa, octb
        if image.ndim == 3 and image.shape[2] not in (1, 3):
            raise ImageValidationError(
                f"{mod_clean.upper()} image must have 1 or 3 channels, got shape {image.shape}."
            )

    # Degeneracy check: Completely uniform flat black (all 0) or all 255
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if min_val == max_val:
        raise ImageValidationError(
            f"Degenerate image: All pixels have identical value {min_val} (zero information content)."
        )
