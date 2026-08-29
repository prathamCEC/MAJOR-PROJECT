"""
Brightness and Illumination Assessment Module for Retinal Images.

Evaluates illumination levels non-destructively:
- Grayscale intensity distribution for OCT-A/OCT-B.
- Perceptual luminance (L-channel in CIE LAB) for Fundus photography.
"""

from typing import Dict
import cv2
import numpy as np


def compute_brightness_metrics(image: np.ndarray, is_color: bool = False) -> Dict[str, float]:
    """
    Calculate raw brightness and illumination metrics.

    Args:
        image: Input retinal image.
        is_color: Set True for Fundus to evaluate luminance channel in LAB space.

    Returns:
        Dictionary containing:
        - 'mean_brightness': Mean pixel intensity/luminance [0.0 - 255.0].
        - 'median_brightness': Median pixel intensity/luminance [0.0 - 255.0].
        - 'p10_brightness': 10th percentile intensity.
        - 'p90_brightness': 90th percentile intensity.
    """
    if image is None or image.size == 0:
        return {
            "mean_brightness": 0.0,
            "median_brightness": 0.0,
            "p10_brightness": 0.0,
            "p90_brightness": 0.0,
        }

    if is_color and image.ndim == 3 and image.shape[2] == 3:
        # Evaluate luminance non-destructively
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0]
    elif image.ndim == 3:
        luminance = image[:, :, 0]
    else:
        luminance = image

    lum_float = luminance.astype(np.float64)

    return {
        "mean_brightness": float(np.mean(lum_float)),
        "median_brightness": float(np.median(lum_float)),
        "p10_brightness": float(np.percentile(lum_float, 10)),
        "p90_brightness": float(np.percentile(lum_float, 90)),
    }
