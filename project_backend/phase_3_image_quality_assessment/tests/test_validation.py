"""
Tests for Phase 3 Validation.
"""

import numpy as np
import pytest

from phase_3_image_quality_assessment.src.validation import (
    ImageValidationError,
    InvalidModalityError,
    validate_modality,
    validate_assessment_image,
)


def test_validate_modality() -> None:
    assert validate_modality("octa") == "octa"
    assert validate_modality("OCTB") == "octb"
    assert validate_modality("Fundus") == "fundus"
    with pytest.raises(InvalidModalityError):
        validate_modality("ct_scan")


def test_validate_assessment_image_valid() -> None:
    img = np.random.randint(20, 220, size=(100, 100, 3), dtype=np.uint8)
    # Should not raise
    validate_assessment_image(img, modality="fundus")


def test_validate_assessment_image_none_raises() -> None:
    with pytest.raises(ImageValidationError, match="None"):
        validate_assessment_image(None, modality="octa")


def test_validate_assessment_image_empty_raises() -> None:
    with pytest.raises(ImageValidationError, match="empty"):
        validate_assessment_image(np.array([], dtype=np.uint8), modality="octa")


def test_validate_assessment_image_nan_inf_raises() -> None:
    img_nan = np.ones((50, 50), dtype=np.float32)
    img_nan[5, 5] = np.nan
    with pytest.raises(ImageValidationError, match="non-finite"):
        validate_assessment_image(img_nan, modality="octa")


def test_validate_assessment_image_degenerate_constant_raises() -> None:
    # All zeros (completely dark)
    dark = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="Degenerate"):
        validate_assessment_image(dark, modality="octa")

    # All white
    white = np.ones((64, 64), dtype=np.uint8) * 255
    with pytest.raises(ImageValidationError, match="Degenerate"):
        validate_assessment_image(white, modality="octa")


def test_validate_fundus_channel_mismatch_raises() -> None:
    # 2D grayscale passed to Fundus validator
    gray = np.random.randint(10, 200, size=(50, 50), dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="Fundus"):
        validate_assessment_image(gray, modality="fundus")
