"""
Tests for Validation Module.
"""

import numpy as np
import pytest

from phase_2_image_preprocessing.src.validation import (
    ImageValidationError,
    InvalidModalityError,
    validate_modality,
    validate_raw_image,
    validate_processed_image,
)


def test_validate_modality_valid() -> None:
    """Test valid modalities."""
    assert validate_modality("octa") == "octa"
    assert validate_modality("OCTB") == "octb"
    assert validate_modality(" Fundus ") == "fundus"


def test_validate_modality_invalid() -> None:
    """Test invalid modality inputs."""
    with pytest.raises(InvalidModalityError):
        validate_modality("xray")

    with pytest.raises(InvalidModalityError):
        validate_modality(123)  # type: ignore


def test_validate_raw_image_valid_grayscale() -> None:
    """Test validating valid raw grayscale image."""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    # Should not raise
    validate_raw_image(img, modality="octa")
    validate_raw_image(img, modality="octb")


def test_validate_raw_image_valid_fundus() -> None:
    """Test validating valid raw fundus image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Should not raise
    validate_raw_image(img, modality="fundus")


def test_validate_raw_image_none_raises() -> None:
    """Test that None image raises ImageValidationError."""
    with pytest.raises(ImageValidationError, match="None"):
        validate_raw_image(None, modality="octa")


def test_validate_raw_image_empty_raises() -> None:
    """Test that 0-sized array raises ImageValidationError."""
    img = np.array([], dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="empty"):
        validate_raw_image(img, modality="octa")


def test_validate_raw_image_nan_inf_raises() -> None:
    """Test that arrays with NaN or Inf raise ImageValidationError."""
    img_nan = np.ones((50, 50), dtype=np.float32)
    img_nan[10, 10] = np.nan
    with pytest.raises(ImageValidationError, match="non-finite"):
        validate_raw_image(img_nan, modality="octa")

    img_inf = np.ones((50, 50), dtype=np.float32)
    img_inf[10, 10] = np.inf
    with pytest.raises(ImageValidationError, match="non-finite"):
        validate_raw_image(img_inf, modality="octa")


def test_validate_raw_image_fundus_channel_mismatch_raises() -> None:
    """Test that single-channel image fails Fundus validation."""
    img_gray = np.ones((50, 50), dtype=np.uint8) * 100
    with pytest.raises(ImageValidationError, match="Fundus"):
        validate_raw_image(img_gray, modality="fundus")


def test_validate_processed_image_valid() -> None:
    """Test valid processed image passing check."""
    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    validate_processed_image(img, target_size=(224, 224), expected_channels=3)


def test_validate_processed_image_dimension_mismatch_raises() -> None:
    """Test processed image with wrong dimensions raises ImageValidationError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="dimensions mismatch"):
        validate_processed_image(img, target_size=(224, 224), expected_channels=3)


def test_validate_processed_image_dtype_mismatch_raises() -> None:
    """Test processed image with float32 instead of uint8 raises ImageValidationError."""
    img = np.zeros((224, 224, 3), dtype=np.float32)
    with pytest.raises(ImageValidationError, match="dtype must be uint8"):
        validate_processed_image(img, target_size=(224, 224), expected_channels=3)
