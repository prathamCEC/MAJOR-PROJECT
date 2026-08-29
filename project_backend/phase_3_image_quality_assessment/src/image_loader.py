"""
Image Loader Module for Phase 3 Retinal Image Quality Assessment.

Loads retinal images non-destructively with cross-platform Unicode path safety,
format checking, and modality-appropriate channel handling.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np

from .config import SUPPORTED_IMAGE_EXTENSIONS
from .validation import (
    validate_modality,
    validate_assessment_image,
    CorruptedImageError,
    ImageValidationError,
)


def load_image(file_path: Union[str, Path], modality: str) -> np.ndarray:
    """
    Load an image for technical quality assessment without modifying its contents.

    Args:
        file_path: Path to the image file.
        modality: Modality identifier ('octa', 'octb', 'fundus').

    Returns:
        np.ndarray containing loaded image data in uint8 format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageValidationError: If the extension is unsupported or validation fails.
        CorruptedImageError: If the image cannot be decoded.
    """
    path = Path(file_path).resolve()
    clean_modality = validate_modality(modality)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ImageValidationError(f"Specified path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{suffix}'. Supported: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    if path.stat().st_size == 0:
        raise CorruptedImageError(f"Image file is 0 bytes (empty): {path}")

    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        # Load with unchanged flags to inspect native channels first
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise CorruptedImageError(f"Failed to read image '{path}': {e}") from e

    if image is None or image.size == 0:
        # Fallback to standard color/grayscale read
        read_flag = cv2.IMREAD_COLOR if clean_modality == "fundus" else cv2.IMREAD_UNCHANGED
        try:
            image = cv2.imread(str(path), read_flag)
        except Exception:
            pass

    if image is None or image.size == 0:
        raise CorruptedImageError(
            f"Failed to decode image from '{path}'. File may be damaged or unsupported."
        )

    # Handle 4-channel RGBA by stripping alpha channel to standard 3-channel BGR
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # If modality is fundus and image is 2D grayscale, raise validation error
    if clean_modality == "fundus" and image.ndim == 2:
        raise ImageValidationError(
            f"Fundus image '{path.name}' is grayscale (2D array), expected 3-channel color."
        )

    # Run validation check
    validate_assessment_image(image, clean_modality)

    return image
