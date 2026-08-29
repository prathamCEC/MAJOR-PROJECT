"""
Safe Black Border Detection and Cropping Module for Retinal Images.

Detects and crops artificial black margins or scanner borders without
destroying anatomical structures (retinal layers, capillaries, optic disc, macula).
If no significant border exists, the original image is preserved untouched.
"""

from typing import Optional
import cv2
import numpy as np


def detect_and_crop_borders(
    image: np.ndarray,
    threshold: int = 10,
    margin: int = 2,
    min_border_ratio: float = 0.02,
    min_content_ratio: float = 0.20,
) -> np.ndarray:
    """
    Detect black borders around the retinal image content and safely crop them.

    Args:
        image: Input retinal image (2D grayscale or 3D color).
        threshold: Pixel intensity threshold to distinguish content from black border.
        margin: Safety margin (pixels) to expand bounding box outward, avoiding clipping content.
        min_border_ratio: Minimum area fraction occupied by borders to trigger a crop (default 2%).
                          If border is smaller than this, image is returned untouched.
        min_content_ratio: Minimum fraction of dimension required for detected content.
                           Prevents collapsing on isolated noise artifacts.

    Returns:
        Cropped numpy.ndarray if meaningful borders exist, otherwise original image.
    """
    if image is None or image.size == 0:
        return image

    height, width = image.shape[:2]

    # Convert to single-channel intensity for mask computation if color
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    # Threshold to identify content pixels
    content_mask = gray > threshold

    # If completely dark or no pixels above threshold, return original
    if not np.any(content_mask):
        return image

    # Compute bounding box along rows and columns
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return image

    ymin, ymax = row_indices[0], row_indices[-1] + 1
    xmin, xmax = col_indices[0], col_indices[-1] + 1

    crop_h = ymax - ymin
    crop_w = xmax - xmin

    # Sanity check: Ensure content region is reasonably sized (avoids tiny speckle artifacts)
    if crop_h < int(height * min_content_ratio) or crop_w < int(width * min_content_ratio):
        return image

    # Check if border is meaningful: if content covers >= (1 - min_border_ratio) of full image, keep original
    content_area = crop_h * crop_w
    total_area = height * width
    border_area_ratio = 1.0 - (content_area / total_area)

    if border_area_ratio < min_border_ratio:
        # Border is negligible, return original image unchanged
        return image

    # Apply safety margin outward to preserve outer vessel/layer boundaries
    ymin = max(0, ymin - margin)
    ymax = min(height, ymax + margin)
    xmin = max(0, xmin - margin)
    xmax = min(width, xmax + margin)

    # Perform slice
    cropped = image[ymin:ymax, xmin:xmax]

    # If cropped result has invalid dimensions, fallback to original
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return image

    return cropped
