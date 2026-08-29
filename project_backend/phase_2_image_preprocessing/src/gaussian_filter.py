"""
Gaussian Filtering Module for Phase 2 Retinal Image Preprocessing.

Provides conservative Gaussian smoothing to reduce high-frequency acquisition noise
while preserving sharp vessel edges, layer boundaries, and macular anatomy.
"""

from typing import Tuple
import cv2
import numpy as np


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Apply conservative Gaussian blur to a retinal image.

    Args:
        image: Input numpy ndarray (2D grayscale or 3D color).
        kernel_size: Gaussian kernel dimensions (width, height) - must be odd positive integers.
        sigma: Gaussian kernel standard deviation in X/Y. If 0, calculated from kernel size.

    Returns:
        Filtered numpy.ndarray with the same shape and data type as input.
    """
    if image is None or image.size == 0:
        return image

    kw, kh = kernel_size
    # Ensure kernel dimensions are positive odd numbers
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1
    kw = max(1, kw)
    kh = max(1, kh)

    return cv2.GaussianBlur(image, (kw, kh), sigmaX=float(sigma), sigmaY=float(sigma))
