"""
Blur and Sharpness Assessment Module for Retinal Images.

Measures spatial high-frequency edge energy using Laplacian variance and
gradient energy metrics to quantify focus and sharpness non-destructively.
"""

from typing import Dict, Union
import cv2
import numpy as np


def compute_blur_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate raw sharpness and blur metrics on an image.

    Args:
        image: Input retinal image (2D grayscale or 3D color).

    Returns:
        Dictionary containing:
        - 'laplacian_variance': Variance of the Laplacian operator.
        - 'tenengrad_gradient_energy': Mean squared Sobel gradient magnitude.
    """
    if image is None or image.size == 0:
        return {"laplacian_variance": 0.0, "tenengrad_gradient_energy": 0.0}

    # Extract 2D intensity non-destructively for gradient calculation
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    # 1. Modified Laplacian Variance
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_var = float(np.var(lap))

    # 2. Tenengrad Gradient Energy (Sobel operators)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x**2 + sobel_y**2))

    return {
        "laplacian_variance": lap_var,
        "tenengrad_gradient_energy": tenengrad,
    }
