"""
Utility Functions for Phase 2 Retinal Image Preprocessing.

Provides helper routines for file discovery, deterministic naming, channel standardization
for Swin Transformer, disk operations, and reload verification.
"""

from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np

from .config import SUPPORTED_IMAGE_EXTENSIONS, TARGET_WIDTH, TARGET_HEIGHT
from .validation import ImageSaveError, ImageValidationError


def find_image_files(directory: Union[str, Path]) -> List[Path]:
    """
    Search a directory recursively or flat for all supported retinal image files.

    Args:
        directory: Directory path to search.

    Returns:
        Sorted list of Path objects for all supported images found.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    found_files: List[Path] = []
    for item in dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            found_files.append(item)

    return sorted(found_files, key=lambda p: p.name.lower())


def get_processed_filename(raw_path: Union[str, Path], output_ext: str = "png") -> str:
    """
    Generate a clean, deterministic filename for the processed image.
    Ensures 'patient001.png' -> 'patient001_processed.png' without duplicating '_processed_processed'.

    Args:
        raw_path: Original image Path or filename string.
        output_ext: Output file extension (default: 'png').

    Returns:
        String filename for the processed artifact.
    """
    stem = Path(raw_path).stem
    ext = output_ext.lstrip(".")

    # Avoid duplicating _processed
    if stem.endswith("_processed"):
        clean_stem = stem
    else:
        clean_stem = f"{stem}_processed"

    return f"{clean_stem}.{ext}"


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    """
    Standardize image representation to 3-channel (H, W, 3).
    Ensures seamless compatibility with downstream Vision Transformer / Swin Transformer models.

    - If 2D (H, W): Replicates single channel across 3 channels via cv2.COLOR_GRAY2BGR.
    - If 3D with 1 channel (H, W, 1): Squeezes and converts to 3-channel.
    - If 3D with 3 channels (H, W, 3): Returns untouched.

    Args:
        image: Input numpy ndarray.

    Returns:
        np.ndarray of shape (H, W, 3).
    """
    if image is None or image.size == 0:
        raise ImageValidationError("Cannot convert empty image to 3 channels.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        if image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            return image
        else:
            raise ImageValidationError(
                f"Unsupported number of channels for 3-channel standardization: {image.shape[2]}"
            )

    raise ImageValidationError(f"Unexpected image shape: {image.shape}")


def verify_saved_image(
    path: Union[str, Path],
    expected_shape: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
    expected_channels: int = 3,
) -> bool:
    """
    Verify that an image written to disk can be successfully reopened and decoded,
    and has the exact expected dimensions and channels.

    Args:
        path: Path to the saved image.
        expected_shape: (width, height) expected dimensions.
        expected_channels: Number of expected color channels (default: 3).

    Returns:
        True if verification passes.

    Raises:
        ImageSaveError: If the image cannot be read back or dimensions mismatch.
    """
    p = Path(path).resolve()
    if not p.exists() or p.stat().st_size == 0:
        raise ImageSaveError(f"Saved file missing or zero size on disk: {p}")

    try:
        raw_bytes = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise ImageSaveError(f"Failed to reopen saved image '{p}': {e}") from e

    if img is None:
        raise ImageSaveError(f"Reopened image is None: corrupted write at '{p}'.")

    exp_w, exp_h = expected_shape
    if img.shape[0] != exp_h or img.shape[1] != exp_w:
        raise ImageSaveError(
            f"Saved image dimension mismatch on disk: expected ({exp_h}, {exp_w}), got ({img.shape[0]}, {img.shape[1]})."
        )

    actual_channels = 1 if img.ndim == 2 else img.shape[2]
    if actual_channels != expected_channels:
        raise ImageSaveError(
            f"Saved image channel mismatch on disk: expected {expected_channels}, got {actual_channels}."
        )

    return True


def save_processed_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    expected_shape: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
) -> Path:
    """
    Save preprocessed image to disk safely and verify it can be reopened.

    Args:
        image: Processed uint8 numpy ndarray.
        output_path: Destination file path.
        expected_shape: (width, height) for verification.

    Returns:
        Resolved Path to the saved image.

    Raises:
        ImageSaveError: If writing or reload verification fails.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ext = out_p.suffix.lower()
    if not ext:
        ext = ".png"
        out_p = out_p.with_suffix(ext)

    # Encode and write using np.tofile for robust Windows path handling
    success, encoded = cv2.imencode(ext, image)
    if not success or encoded is None:
        raise ImageSaveError(f"Failed to encode image to format '{ext}' for '{out_p}'.")

    try:
        encoded.tofile(str(out_p))
    except Exception as e:
        raise ImageSaveError(f"Failed to write image bytes to '{out_p}': {e}") from e

    # Immediately verify disk reload
    channels = 3 if image.ndim == 3 else 1
    verify_saved_image(out_p, expected_shape=expected_shape, expected_channels=channels)

    return out_p
