"""
Tests for Phase 3 Image Loader.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from phase_3_image_quality_assessment.src.image_loader import load_image
from phase_3_image_quality_assessment.src.validation import (
    CorruptedImageError,
    ImageValidationError,
    InvalidModalityError,
)


@pytest.fixture
def test_images(tmp_path: Path) -> Path:
    """Fixture creating test images in multiple formats."""
    # 1. Grayscale PNG (OCT)
    gray = np.random.randint(40, 200, size=(120, 120), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "oct.png"), gray)

    # 2. Color PNG (Fundus)
    color = np.random.randint(40, 200, size=(120, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "fundus.png"), color)

    # 3. 4-Channel RGBA
    rgba = np.random.randint(40, 200, size=(100, 100, 4), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "rgba.png"), rgba)

    # 4. JPEG
    cv2.imwrite(str(tmp_path / "fundus.jpg"), color)

    # 5. TIFF
    cv2.imwrite(str(tmp_path / "oct.tif"), gray)

    # 6. Corrupted file
    (tmp_path / "corrupt.png").write_bytes(b"CORRUPT_HEADER_BYTES_123")

    # 7. 0-byte file
    (tmp_path / "empty.png").touch()

    return tmp_path


def test_load_grayscale_oct(test_images: Path) -> None:
    img = load_image(test_images / "oct.png", modality="octa")
    assert isinstance(img, np.ndarray)
    assert img.ndim in (2, 3)
    assert img.dtype == np.uint8


def test_load_color_fundus(test_images: Path) -> None:
    img = load_image(test_images / "fundus.png", modality="fundus")
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_load_rgba_strips_alpha(test_images: Path) -> None:
    img = load_image(test_images / "rgba.png", modality="fundus")
    assert img.shape[2] == 3


def test_load_tiff_format(test_images: Path) -> None:
    img = load_image(test_images / "oct.tif", modality="octb")
    assert isinstance(img, np.ndarray)


def test_load_missing_file_raises(test_images: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(test_images / "nonexistent.png", modality="octa")


def test_load_corrupted_file_raises(test_images: Path) -> None:
    with pytest.raises(CorruptedImageError):
        load_image(test_images / "corrupt.png", modality="octa")


def test_load_empty_file_raises(test_images: Path) -> None:
    with pytest.raises(CorruptedImageError):
        load_image(test_images / "empty.png", modality="octa")


def test_load_invalid_modality_raises(test_images: Path) -> None:
    with pytest.raises(InvalidModalityError):
        load_image(test_images / "oct.png", modality="xray")
