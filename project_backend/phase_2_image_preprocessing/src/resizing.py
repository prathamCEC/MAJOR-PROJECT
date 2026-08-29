"""
Aspect-Ratio-Preserving Resizing and Padding Module for Retinal Images.

Ensures distortion-free dimensional standardization to the target Swin Transformer
input size (default 224 x 224) using aspect-ratio scaling and symmetric padding.
"""

from typing import Tuple, Union
import cv2
import numpy as np

from .config import TARGET_WIDTH, TARGET_HEIGHT


def resize_with_aspect_ratio_and_pad(
    image: np.ndarray,
    target_size: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
    pad_value: Union[int, Tuple[int, int, int]] = 0,
    interpolation_down: int = cv2.INTER_AREA,
    interpolation_up: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """
    Resize image preserving original aspect ratio, then symmetrically pad to target_size.

    Args:
        image: Input numpy ndarray (H, W) or (H, W, C).
        target_size: Desired (target_width, target_height).
        pad_value: Constant value used for letterbox padding borders.
        interpolation_down: Interpolation method when shrinking image.
        interpolation_up: Interpolation method when enlarging image.

    Returns:
        np.ndarray with exact dimensions (target_height, target_width) or (target_height, target_width, C).
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot resize empty or None image.")

    target_w, target_h = target_size
    orig_h, orig_w = image.shape[:2]

    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid original dimensions: ({orig_h}, {orig_w})")

    # If already exact target size, return copy
    if orig_w == target_w and orig_h == target_h:
        return image.copy()

    # Calculate aspect-ratio-preserving scaling factor
    scale = min(target_w / float(orig_w), target_h / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Choose interpolation based on whether we are downscaling or upscaling
    if scale < 1.0:
        interp = interpolation_down
    else:
        interp = interpolation_up

    # Perform aspect-ratio-preserving resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Compute symmetric padding margins
    pad_total_w = target_w - new_w
    pad_total_h = target_h - new_h

    pad_top = pad_total_h // 2
    pad_bottom = pad_total_h - pad_top
    pad_left = pad_total_w // 2
    pad_right = pad_total_w - pad_left

    # Construct borders using cv2.copyMakeBorder
    if isinstance(pad_value, (tuple, list)):
        value = list(pad_value)
    else:
        value = [int(pad_value)]

    padded = cv2.copyMakeBorder(
        resized,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=value,
    )

    # Ensure output has exact target height and width
    padded = padded[:target_h, :target_w]

    return padded
