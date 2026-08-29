"""
Tests for Noise Metric Extraction.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.noise import compute_noise_metrics


def test_clean_vs_noisy_image() -> None:
    # Smooth clean image
    clean = np.ones((80, 80), dtype=np.uint8) * 120

    # Add Gaussian noise
    noise = np.random.normal(0, 25, (80, 80))
    noisy = np.clip(clean.astype(float) + noise, 0, 255).astype(np.uint8)

    clean_res = compute_noise_metrics(clean)
    noisy_res = compute_noise_metrics(noisy)

    assert noisy_res["noise_residual_std"] > clean_res["noise_residual_std"]
