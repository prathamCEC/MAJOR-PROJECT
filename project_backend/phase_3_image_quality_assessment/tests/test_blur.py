"""
Tests for Sharpness and Blur Metric Extraction.
"""

import cv2
import numpy as np
import pytest

from phase_3_image_quality_assessment.src.blur_detection import compute_blur_metrics


def test_sharp_vs_blurred_image() -> None:
    """Test that a sharp patterned image has significantly higher Laplacian variance than blurred."""
    # Create sharp checkerboard pattern
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[::10, :] = 255
    sharp[:, ::10] = 255

    # Apply heavy blur
    blurred = cv2.GaussianBlur(sharp, (15, 15), sigmaX=5.0)

    sharp_metrics = compute_blur_metrics(sharp)
    blurred_metrics = compute_blur_metrics(blurred)

    assert sharp_metrics["laplacian_variance"] > blurred_metrics["laplacian_variance"]
    assert sharp_metrics["tenengrad_gradient_energy"] > blurred_metrics["tenengrad_gradient_energy"]
