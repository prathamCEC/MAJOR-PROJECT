"""
Image Loading Module for Phase 2 Retinal Image Preprocessing.

Provides robust, modality-aware image loading using OpenCV with cross-platform
file path handling, corruption detection, and format validation.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np

from .config import SUPPORTED_IMAGE_EXTENSIONS
from .validation import (
    validate_modality,
    validate_raw_image,
    CorruptedImageError,
    ImageValidationError,
)


def load_image(file_path: Union[str, Path], modality: str) -> np.ndarray:
    """
    Load an image from disk in a modality-aware manner.

    - For 'fundus': Loads as 3-channel color (BGR).
    - For 'octa' / 'octb': Loads as single-channel grayscale.

    Args:
        file_path: Path to the image file (string or pathlib.Path).
        modality: Imaging modality identifier ('octa', 'octb', 'fundus').

    Returns:
        np.ndarray containing loaded image data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageValidationError: If file extension is unsupported or validation fails.
        CorruptedImageError: If the file cannot be decoded or is empty.
    """
    path = Path(file_path).resolve()
    clean_modality = validate_modality(modality)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ImageValidationError(f"Specified path is not a file: {path}")

    # Check extension
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{suffix}'. Supported: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    # Check file size
    if path.stat().st_size == 0:
        raise CorruptedImageError(f"Image file is 0 bytes (empty): {path}")

    # Read image using cv2.imdecode with np.fromfile for robust cross-platform path handling
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        if clean_modality == "fundus":
            read_flag = cv2.IMREAD_COLOR
        else:
            read_flag = cv2.IMREAD_GRAYSCALE

        image = cv2.imdecode(raw_bytes, read_flag)
    except Exception as e:
        raise CorruptedImageError(f"Failed to read image file '{path}': {e}") from e

    if image is None or image.size == 0:
        # Fallback to direct cv2.imread if imdecode returned None
        try:
            image = cv2.imread(str(path), read_flag)
        except Exception:
            pass

    if image is None or image.size == 0:
        raise CorruptedImageError(
            f"Failed to decode image from '{path}'. File may be corrupted or format unreadable."
        )

    # Run validation immediately
    validate_raw_image(image, clean_modality)

    return image
