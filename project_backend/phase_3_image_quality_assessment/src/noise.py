"""
Noise Assessment Module for Retinal Images.

Estimates high-frequency acquisition noise using non-destructive residual analysis.
"""

from typing import Dict
import cv2
import numpy as np


def compute_noise_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Estimate image noise level via high-frequency spatial residual analysis.

    Args:
        image: Input retinal image.

    Returns:
        Dictionary containing:
        - 'noise_residual_std': Standard deviation of the high-frequency residual.
        - 'estimated_snr_db': Estimated signal-to-noise ratio in decibels.
    """
    if image is None or image.size == 0:
        return {"noise_residual_std": 0.0, "estimated_snr_db": 0.0}

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    gray_float = gray.astype(np.float64)

    # Smooth with conservative filter to isolate low-frequency signal
    blurred = cv2.GaussianBlur(gray_float, (3, 3), sigmaX=1.0, sigmaY=1.0)

    # Residual high-frequency component
    residual = gray_float - blurred

    noise_std = float(np.std(residual))
    signal_mean = float(np.mean(gray_float))

    if noise_std > 1e-4 and signal_mean > 1e-4:
        snr_db = float(20.0 * np.log10(signal_mean / noise_std))
    else:
        snr_db = 0.0

    return {
        "noise_residual_std": noise_std,
        "estimated_snr_db": max(0.0, snr_db),
    }
