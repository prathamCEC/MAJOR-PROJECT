"""
Tests for Image Loader Module.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from phase_2_image_preprocessing.src.image_loader import load_image
from phase_2_image_preprocessing.src.validation import (
    CorruptedImageError,
    ImageValidationError,
    InvalidModalityError,
)


@pytest.fixture
def temp_images_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory with synthetic test images."""
    # 1. Valid Grayscale Image (OCT-A/B fixture)
    gray_img = np.random.randint(20, 230, size=(100, 100), dtype=np.uint8)
    gray_path = tmp_path / "test_oct.png"
    cv2.imwrite(str(gray_path), gray_img)

    # 2. Valid Color Image (Fundus fixture)
    color_img = np.random.randint(20, 230, size=(120, 160, 3), dtype=np.uint8)
    color_path = tmp_path / "test_fundus.png"
    cv2.imwrite(str(color_path), color_img)

    # 3. Valid TIFF image
    tif_path = tmp_path / "test_image.tif"
    cv2.imwrite(str(tif_path), gray_img)

    # 4. Valid JPEG image
    jpg_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(jpg_path), color_img)

    # 5. Empty (0-byte) file
    empty_path = tmp_path / "empty.png"
    empty_path.touch()

    # 6. Corrupted file (invalid header bytes)
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_BYTES_12345")

    # 7. Unsupported extension
    unsupported_path = tmp_path / "sample.txt"
    unsupported_path.write_text("dummy text")

    return tmp_path


def test_load_valid_grayscale_oct(temp_images_dir: Path) -> None:
    """Test loading valid grayscale OCT image."""
    img_path = temp_images_dir / "test_oct.png"
    img = load_image(img_path, modality="octa")
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert img.shape == (100, 100)
    assert img.dtype == np.uint8


def test_load_valid_color_fundus(temp_images_dir: Path) -> None:
    """Test loading valid color Fundus image."""
    img_path = temp_images_dir / "test_fundus.png"
    img = load_image(img_path, modality="fundus")
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape == (120, 160, 3)
    assert img.dtype == np.uint8


def test_load_tiff_format(temp_images_dir: Path) -> None:
    """Test loading TIFF image."""
    img_path = temp_images_dir / "test_image.tif"
    img = load_image(img_path, modality="octb")
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100)


def test_load_jpeg_format(temp_images_dir: Path) -> None:
    """Test loading JPEG image."""
    img_path = temp_images_dir / "test_image.jpg"
    img = load_image(img_path, modality="fundus")
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3


def test_load_missing_file_raises(temp_images_dir: Path) -> None:
    """Test that a non-existent file path raises FileNotFoundError."""
    missing_path = temp_images_dir / "non_existent_file.png"
    with pytest.raises(FileNotFoundError):
        load_image(missing_path, modality="octa")


def test_load_empty_file_raises(temp_images_dir: Path) -> None:
    """Test that a 0-byte file raises CorruptedImageError."""
    empty_path = temp_images_dir / "empty.png"
    with pytest.raises(CorruptedImageError):
        load_image(empty_path, modality="octa")


def test_load_corrupted_file_raises(temp_images_dir: Path) -> None:
    """Test that a corrupted header file raises CorruptedImageError."""
    corrupt_path = temp_images_dir / "corrupt.png"
    with pytest.raises(CorruptedImageError):
        load_image(corrupt_path, modality="octa")


def test_load_unsupported_extension_raises(temp_images_dir: Path) -> None:
    """Test that an unsupported extension raises ImageValidationError."""
    unsupported_path = temp_images_dir / "sample.txt"
    with pytest.raises(ImageValidationError):
        load_image(unsupported_path, modality="octa")


def test_load_invalid_modality_raises(temp_images_dir: Path) -> None:
    """Test that specifying an invalid modality raises InvalidModalityError."""
    img_path = temp_images_dir / "test_oct.png"
    with pytest.raises(InvalidModalityError):
        load_image(img_path, modality="mri")
