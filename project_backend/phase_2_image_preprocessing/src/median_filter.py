"""
Median Filtering Module for Phase 2 Retinal Image Preprocessing.

Provides conservative non-linear median filtering for impulse and speckle noise
reduction in structural imaging (e.g. OCT-B) while preserving sharp anatomical boundaries.
"""

import cv2
import numpy as np


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply conservative median blur to a retinal image.

    Args:
        image: Input numpy ndarray (uint8 2D grayscale or 3D color).
        kernel_size: Aperture linear size (must be an odd integer >= 3).

    Returns:
        Filtered numpy.ndarray with preserved edges.
    """
    if image is None or image.size == 0:
        return image

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    k = max(1, k)

    if k == 1:
        return image

    return cv2.medianBlur(image, k)
