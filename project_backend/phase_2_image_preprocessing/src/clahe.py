"""
Contrast Limited Adaptive Histogram Equalization (CLAHE) Module.

Applies modality-appropriate CLAHE contrast enhancement:
- OCT-A / OCT-B: Single-channel grayscale intensity CLAHE.
- Fundus: Luminance-channel CLAHE in LAB color space to preserve diagnostic color integrity.
"""

from typing import Tuple
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    is_color: bool = False,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Args:
        image: Input retinal image (uint8 numpy ndarray).
        clip_limit: Threshold for contrast limiting (default: 2.0).
        tile_grid_size: Grid dimensions for localized histogram equalization (default: (8, 8)).
        is_color: Set True for color images (Fundus) to use LAB luminance equalization.

    Returns:
        np.ndarray with enhanced contrast matching original input dtype and shape.
    """
    if image is None or image.size == 0:
        return image

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )

    # Grayscale image (OCT-A / OCT-B or 2D array)
    if not is_color or image.ndim == 2:
        if image.ndim == 2:
            return clahe.apply(image)
        elif image.ndim == 3 and image.shape[2] == 1:
            equalized = clahe.apply(image[:, :, 0])
            return equalized[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[2] == 3 and not is_color:
            # Grayscale duplicated to 3 channels: apply to one channel and replicate
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eq_gray = clahe.apply(gray)
            return cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2BGR)

    # Color Fundus image: Convert BGR -> LAB, apply CLAHE to L-channel, convert back to BGR
    if is_color and image.ndim == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_clahe = clahe.apply(l_chan)
        lab_clahe = cv2.merge((l_clahe, a_chan, b_chan))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return image
