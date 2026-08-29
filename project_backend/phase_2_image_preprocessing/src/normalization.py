"""
Pixel Normalization and Type Conversion Module for Retinal Images.

Manages clean transitions between:
- Internal mathematical processing (float32 [0.0, 1.0])
- Output image representation (uint8 [0, 255])

Ensures no loss of dynamic range and prevents near-black image saving bugs.
"""

from typing import Tuple
import numpy as np


def normalize_to_float32(
    image: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """
    Convert image to float32 normalized in [target_min, target_max].

    Args:
        image: Input image (uint8, uint16, or float numpy ndarray).
        target_min: Lower bound for output float range (default: 0.0).
        target_max: Upper bound for output float range (default: 1.0).

    Returns:
        np.ndarray with dtype float32 scaled to [target_min, target_max].
    """
    if image is None or image.size == 0:
        return image

    img_float = image.astype(np.float32)

    if image.dtype == np.uint8:
        norm = img_float / 255.0
    elif image.dtype == np.uint16:
        norm = img_float / 65535.0
    else:
        # For arbitrary float input, perform robust min-max scaling
        min_v = float(np.min(img_float))
        max_v = float(np.max(img_float))
        if max_v > min_v:
            norm = (img_float - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(img_float)

    if target_min != 0.0 or target_max != 1.0:
        norm = norm * (target_max - target_min) + target_min

    return norm.astype(np.float32)


def convert_to_uint8(
    image: np.ndarray,
    source_min: float = 0.0,
    source_max: float = 1.0,
) -> np.ndarray:
    """
    Convert float32 image in [source_min, source_max] back to standard uint8 [0, 255].

    Args:
        image: Input numpy ndarray (float32, float64, or uint8).
        source_min: Expected minimum value of float input.
        source_max: Expected maximum value of float input.

    Returns:
        np.ndarray with dtype uint8 strictly bounded in [0, 255].
    """
    if image is None or image.size == 0:
        return image

    if image.dtype == np.uint8:
        return image.copy()

    img_float = image.astype(np.float32)

    # If already in [0, 255] range for floats
    if np.max(img_float) > 1.0:
        clipped = np.clip(img_float, 0.0, 255.0)
        return np.round(clipped).astype(np.uint8)

    # Scale from [source_min, source_max] to [0, 255]
    diff = source_max - source_min
    if diff <= 0:
        diff = 1.0

    scaled = ((img_float - source_min) / diff) * 255.0
    clipped = np.clip(scaled, 0.0, 255.0)
    return np.round(clipped).astype(np.uint8)
