"""
Color Quality Assessment Module for Fundus Photography.

Evaluates technical color integrity, saturation dynamics, and chromatic balance
without enforcing artificial color equality across natural biological tissue.
"""

from typing import Dict
import cv2
import numpy as np


def compute_color_quality_metrics(image: np.ndarray, is_color: bool = True) -> Dict[str, float]:
    """
    Calculate color fidelity and chromatic distribution metrics on Fundus images.

    Args:
        image: Input retinal image (3D BGR array).
        is_color: Set True for Fundus, False for grayscale OCT.

    Returns:
        Dictionary containing:
        - 'is_color_valid': 1.0 if true color representation, 0.0 if monochrome/degenerate.
        - 'mean_saturation': Mean saturation value in HSV space [0.0 - 255.0].
        - 'color_channel_disparity': Inter-channel difference metric.
        - 'color_cast_ratio': Ratio measuring excessive single-channel dominance.
    """
    if not is_color or image is None or image.ndim != 3 or image.shape[2] != 3:
        return {
            "is_color_valid": 1.0,
            "mean_saturation": 0.0,
            "color_channel_disparity": 0.0,
            "color_cast_ratio": 0.0,
        }

    b_chan = image[:, :, 0].astype(np.float64)
    g_chan = image[:, :, 1].astype(np.float64)
    r_chan = image[:, :, 2].astype(np.float64)

    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float64)

    # Exclude outer dark background when computing saturation
    mask = (b_chan + g_chan + r_chan) > 30.0
    if np.count_nonzero(mask) > 100:
        active_sat = sat[mask]
        mean_sat = float(np.mean(active_sat))
    else:
        mean_sat = float(np.mean(sat))

    # Inter-channel difference (detects fake monochrome/grayscale 3-channel duplicate)
    bg_diff = np.mean(np.abs(b_chan - g_chan))
    gr_diff = np.mean(np.abs(g_chan - r_chan))
    disparity = float(bg_diff + gr_diff)

    # If all channels are identical (disparity ~ 0), image is actually grayscale, not real color
    is_valid = 1.0 if disparity > 2.0 else 0.0

    # Channel dominance / severe color cast check
    mean_b = float(np.mean(b_chan)) + 1e-4
    mean_g = float(np.mean(g_chan)) + 1e-4
    mean_r = float(np.mean(r_chan)) + 1e-4
    total_mean = mean_b + mean_g + mean_r
    max_dominance = max(mean_b, mean_g, mean_r) / total_mean  # Normal fundus has high R, but not 0.99

    return {
        "is_color_valid": is_valid,
        "mean_saturation": mean_sat,
        "color_channel_disparity": disparity,
        "color_cast_ratio": float(max_dominance),
    }
