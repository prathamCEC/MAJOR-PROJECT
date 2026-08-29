"""
Metric Normalization Module for Phase 3 Retinal Image Quality Assessment.

Calibrates raw technical metrics onto a standardized 0 - 100 quality scale
(100 = optimal technical quality, 0 = severe technical failure) using modality-specific curves.
"""

import math
from typing import Any, Dict
import numpy as np

from .config import ModalityQualityConfig


def normalize_blur(raw_lap_var: float, config: ModalityQualityConfig) -> float:
    """
    Map raw Laplacian variance to a 0-100 sharpness score.

    Uses logarithmic scaling to handle large dynamic range in Laplacian variances.
    """
    if raw_lap_var <= 0.0:
        return 0.0

    min_ref = max(1.0, config.blur_raw_min)
    max_ref = max(min_ref + 1.0, config.blur_raw_max)

    log_val = math.log10(max(1.0, raw_lap_var))
    log_min = math.log10(min_ref)
    log_max = math.log10(max_ref)

    if log_val <= log_min:
        score = (raw_lap_var / min_ref) * 40.0
    elif log_val >= log_max:
        score = 100.0
    else:
        fraction = (log_val - log_min) / (log_max - log_min)
        score = 40.0 + (fraction * 60.0)

    return float(np.clip(score, 0.0, 100.0))


def normalize_brightness(mean_brightness: float, config: ModalityQualityConfig) -> float:
    """
    Map mean brightness to a 0-100 illumination score using a trapezoidal curve.
    """
    b = float(mean_brightness)
    crit_low = config.brightness_crit_low
    opt_low = config.brightness_opt_low
    opt_high = config.brightness_opt_high
    crit_high = config.brightness_crit_high

    if b <= crit_low or b >= crit_high:
        return 0.0
    elif opt_low <= b <= opt_high:
        return 100.0
    elif crit_low < b < opt_low:
        return float(((b - crit_low) / (opt_low - crit_low)) * 100.0)
    else:  # opt_high < b < crit_high
        return float(((crit_high - b) / (crit_high - opt_high)) * 100.0)


def normalize_contrast(rms_contrast: float, config: ModalityQualityConfig) -> float:
    """
    Map RMS contrast to a 0-100 score.
    """
    c = float(rms_contrast)
    crit_low = config.contrast_crit_low
    opt_low = config.contrast_opt_low
    opt_high = config.contrast_opt_high

    if c <= crit_low:
        return float(max(0.0, (c / crit_low) * 30.0))
    elif opt_low <= c <= opt_high:
        return 100.0
    elif crit_low < c < opt_low:
        return float(30.0 + ((c - crit_low) / (opt_low - crit_low)) * 70.0)
    else:  # c > opt_high (excessive contrast/sharp clipping)
        penalty = min(40.0, (c - opt_high) * 0.8)
        return float(max(60.0, 100.0 - penalty))


def normalize_noise(noise_std: float, config: ModalityQualityConfig) -> float:
    """
    Map residual noise standard deviation to a 0-100 score (higher score = cleaner image).
    """
    n = float(noise_std)
    clean_max = config.noise_clean_max
    severe_thresh = config.noise_severe_thresh

    if n <= clean_max:
        return 100.0
    elif n >= severe_thresh:
        return 0.0
    else:
        fraction = (severe_thresh - n) / (severe_thresh - clean_max)
        return float(fraction * 100.0)


def normalize_clipping(total_clipping_ratio: float, config: ModalityQualityConfig) -> float:
    """
    Map total clipping fraction to a 0-100 score (higher score = minimal clipping).
    """
    clip = float(total_clipping_ratio)
    clean_max = config.clipping_clean_max
    severe_thresh = config.clipping_severe_thresh

    if clip <= clean_max:
        return 100.0
    elif clip >= severe_thresh:
        return 0.0
    else:
        fraction = (severe_thresh - clip) / (severe_thresh - clean_max)
        return float(fraction * 100.0)


def normalize_content(entropy: float, config: ModalityQualityConfig) -> float:
    """
    Map Shannon entropy (bits) to a 0-100 content score.
    """
    e = float(entropy)
    min_ref = config.entropy_min_ref
    opt_ref = config.entropy_opt_ref

    if e <= min_ref:
        return float(max(0.0, (e / min_ref) * 40.0))
    elif e >= opt_ref:
        return 100.0
    else:
        fraction = (e - min_ref) / (opt_ref - min_ref)
        return float(40.0 + (fraction * 60.0))


def normalize_color(color_metrics: Dict[str, float], config: ModalityQualityConfig) -> float:
    """
    Map Fundus color fidelity and saturation to a 0-100 score.
    """
    if not config.is_color:
        return 100.0

    is_valid = color_metrics.get("is_color_valid", 1.0)
    if is_valid < 0.5:
        # Monochrome or degenerate color in Fundus
        return 20.0

    mean_sat = color_metrics.get("mean_saturation", 0.0)
    cast_ratio = color_metrics.get("color_cast_ratio", 0.0)

    # Saturation score (good Fundus has mean saturation >= 20)
    sat_score = min(100.0, (mean_sat / 30.0) * 100.0)

    # Color cast penalty if single channel exceeds 0.85 of total energy
    cast_penalty = max(0.0, (cast_ratio - 0.75) * 200.0) if cast_ratio > 0.75 else 0.0

    final_color_score = max(0.0, sat_score - cast_penalty)
    return float(np.clip(final_color_score, 0.0, 100.0))


def normalize_all_metrics(
    raw_metrics: Dict[str, Any],
    config: ModalityQualityConfig,
) -> Dict[str, float]:
    """
    Convert all raw metric measurements into standardized 0 - 100 quality scores.

    Args:
        raw_metrics: Dictionary of raw outputs from all detectors.
        config: ModalityQualityConfig object.

    Returns:
        Dictionary mapping each metric name to a float score in [0.0, 100.0].
    """
    blur_raw = raw_metrics.get("laplacian_variance", 0.0)
    brightness_raw = raw_metrics.get("mean_brightness", 0.0)
    contrast_raw = raw_metrics.get("rms_contrast", 0.0)
    noise_raw = raw_metrics.get("noise_residual_std", 0.0)
    clipping_raw = raw_metrics.get("total_clipping_ratio", 0.0)
    entropy_raw = raw_metrics.get("shannon_entropy", 0.0)
    color_dict = raw_metrics.get("color_metrics", {})

    scores = {
        "blur_score": normalize_blur(blur_raw, config),
        "brightness_score": normalize_brightness(brightness_raw, config),
        "contrast_score": normalize_contrast(contrast_raw, config),
        "noise_score": normalize_noise(noise_raw, config),
        "clipping_score": normalize_clipping(clipping_raw, config),
        "content_score": normalize_content(entropy_raw, config),
        "color_score": normalize_color(color_dict, config) if config.is_color else 100.0,
    }

    return scores
