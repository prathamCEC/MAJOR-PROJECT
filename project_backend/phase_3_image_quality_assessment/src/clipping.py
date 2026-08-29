"""
Clipping and Saturation Assessment Module for Retinal Images.

Detects anatomical under-clipping (dark shadows/underexposure) and over-clipping
(sensor saturation/glare) while safely excluding natural black margins in Fundus images.
"""

from typing import Dict
import cv2
import numpy as np


def compute_clipping_metrics(
    image: np.ndarray,
    low_thresh: int = 2,
    high_thresh: int = 253,
    is_color: bool = False,
) -> Dict[str, float]:
    """
    Calculate fraction of clipped/saturated pixels across active image content.

    Args:
        image: Input retinal image.
        low_thresh: Pixel intensity threshold below which pixels are under-clipped.
        high_thresh: Pixel intensity threshold above which pixels are over-clipped.
        is_color: If True (Fundus), excludes natural black background when assessing clipping.

    Returns:
        Dictionary containing:
        - 'underexposed_clipping_ratio': Fraction of content pixels <= low_thresh.
        - 'overexposed_clipping_ratio': Fraction of content pixels >= high_thresh.
        - 'total_clipping_ratio': Sum of clipped fraction.
    """
    if image is None or image.size == 0:
        return {
            "underexposed_clipping_ratio": 0.0,
            "overexposed_clipping_ratio": 0.0,
            "total_clipping_ratio": 0.0,
        }

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    # Exclude natural black border / letterbox padding when assessing clinical clipping
    content_mask = gray > 10
    if np.count_nonzero(content_mask) > (0.05 * gray.size):
        eval_pixels = gray[content_mask]
    else:
        eval_pixels = gray.flatten()

    total_count = max(1, eval_pixels.size)

    under_clipped = np.count_nonzero(eval_pixels <= low_thresh)
    over_clipped = np.count_nonzero(eval_pixels >= high_thresh)

    under_ratio = float(under_clipped / total_count)
    over_ratio = float(over_clipped / total_count)
    total_ratio = float(under_ratio + over_ratio)

    return {
        "underexposed_clipping_ratio": under_ratio,
        "overexposed_clipping_ratio": over_ratio,
        "total_clipping_ratio": total_ratio,
    }
