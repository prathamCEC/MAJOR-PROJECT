"""
Content and Information Integrity Module for Retinal Images.

Computes Shannon information entropy and structural foreground ratio to verify
that an image contains sufficient diagnostic pattern information.
"""

from typing import Dict
import cv2
import numpy as np


def compute_content_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate information entropy and content coverage metrics.

    Args:
        image: Input retinal image (2D grayscale or 3D color).

    Returns:
        Dictionary containing:
        - 'shannon_entropy': Information entropy in bits [0.0 - 8.0].
        - 'foreground_content_ratio': Fraction of active foreground content pixels.
        - 'is_content_sufficient': 1.0 if entropy and coverage exceed minimal technical threshold.
    """
    if image is None or image.size == 0:
        return {
            "shannon_entropy": 0.0,
            "foreground_content_ratio": 0.0,
            "is_content_sufficient": 0.0,
        }

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    # Compute 256-bin histogram
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_norm = hist / float(gray.size)

    # Shannon entropy: -sum(p * log2(p)) for non-zero probabilities
    non_zero = hist_norm[hist_norm > 0]
    entropy = -float(np.sum(non_zero * np.log2(non_zero)))

    # Foreground content ratio (pixels above background threshold of 10)
    foreground_pixels = np.count_nonzero(gray > 10)
    foreground_ratio = float(foreground_pixels / gray.size)

    # Content is sufficient if entropy > 2.0 and foreground ratio > 0.05
    is_sufficient = 1.0 if (entropy > 2.0 and foreground_ratio > 0.05) else 0.0

    return {
        "shannon_entropy": entropy,
        "foreground_content_ratio": foreground_ratio,
        "is_content_sufficient": is_sufficient,
    }
