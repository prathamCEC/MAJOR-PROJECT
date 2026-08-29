"""
Contrast Assessment Module for Retinal Images.

Quantifies intra-image contrast and dynamic range variation across retinal structures.
"""

from typing import Dict
import cv2
import numpy as np


def compute_contrast_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate raw contrast metrics on a retinal image.

    Args:
        image: Input retinal image (2D grayscale or 3D color).

    Returns:
        Dictionary containing:
        - 'rms_contrast': Root-mean-square contrast (standard deviation).
        - 'dynamic_range': Robust 5th-to-95th percentile intensity span.
        - 'michelson_contrast': Global Michelson contrast ratio [0.0 - 1.0].
    """
    if image is None or image.size == 0:
        return {
            "rms_contrast": 0.0,
            "dynamic_range": 0.0,
            "michelson_contrast": 0.0,
        }

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    gray_float = gray.astype(np.float64)

    rms = float(np.std(gray_float))
    p5 = float(np.percentile(gray_float, 5))
    p95 = float(np.percentile(gray_float, 95))
    dyn_range = max(0.0, p95 - p5)

    max_v = float(np.max(gray_float))
    min_v = float(np.min(gray_float))
    sum_v = max_v + min_v
    michelson = (max_v - min_v) / sum_v if sum_v > 0 else 0.0

    return {
        "rms_contrast": rms,
        "dynamic_range": dyn_range,
        "michelson_contrast": michelson,
    }
