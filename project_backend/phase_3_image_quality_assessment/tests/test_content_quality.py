"""
Tests for Content Integrity and Shannon Entropy.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.content_quality import compute_content_metrics


def test_informative_vs_flat_image() -> None:
    # Informative image with varied textures
    informative = np.random.randint(20, 230, size=(100, 100), dtype=np.uint8)
    # Low-information near-flat image
    flat = np.ones((100, 100), dtype=np.uint8) * 128
    flat[0, 0] = 129  # tiny variation to avoid divide by zero

    inf_res = compute_content_metrics(informative)
    flat_res = compute_content_metrics(flat)

    assert inf_res["shannon_entropy"] > flat_res["shannon_entropy"]
    assert inf_res["is_content_sufficient"] == 1.0
    assert flat_res["is_content_sufficient"] == 0.0
