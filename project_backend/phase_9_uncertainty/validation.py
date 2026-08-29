"""
Validation and Numerical Safety Module for Phase 9 Uncertainty Estimation.

Performs input shape validation, NaN/Inf detection, MC sample count checks,
and output statistical bounds verification.
"""

from typing import Any, Dict
import torch

from .config import UncertaintyConfig


def validate_uncertainty_inputs(
    upr: torch.Tensor,
    config: UncertaintyConfig,
    mc_samples: int,
) -> None:
    """
    Validate input representation tensor and MC parameters prior to forward sampling.
    """
    if not isinstance(upr, torch.Tensor):
        raise TypeError(f"UPR input must be a torch.Tensor, got {type(upr)}.")

    if upr.ndim not in (2, 3):
        raise ValueError(f"Expected UPR tensor of rank 2 ([B, 512]) or 3 ([B, 1, 512]), got rank {upr.ndim}.")

    if upr.shape[-1] != 512:
        raise ValueError(f"UPR dimension mismatch: expected 512, received {upr.shape[-1]}.")

    if torch.isnan(upr).any():
        raise ValueError("Unified Patient Representation tensor contains NaN values.")
    if torch.isinf(upr).any():
        raise ValueError("Unified Patient Representation tensor contains infinite (Inf) values.")

    if mc_samples < 2:
        raise ValueError(
            f"Invalid mc_samples={mc_samples}. Monte Carlo Dropout requires at least 2 stochastic passes."
        )


def validate_uncertainty_outputs(
    results: Dict[str, Any],
    expected_batch_size: int,
) -> None:
    """
    Audit statistical bounds and finite validity of generated uncertainty metrics.
    """
    for task in ("stroke", "alzheimer"):
        if task not in results:
            raise KeyError(f"Missing required task result key: '{task}'.")

        task_data = results[task]
        p_mean = task_data["mc_mean_probability"]
        var = task_data["mc_variance"]
        std = task_data["mc_std"]
        entropy = task_data["predictive_entropy"]
        conf = task_data["confidence"]
        conf_pct = task_data["confidence_percent"]
        pred = task_data["prediction"]

        # 1. Shape check (if tensor)
        if isinstance(p_mean, torch.Tensor):
            if p_mean.shape[0] != expected_batch_size:
                raise ValueError(f"Batch size mismatch for {task}: expected {expected_batch_size}, got {p_mean.shape[0]}.")

            if not (0.0 <= p_mean.min().item() and p_mean.max().item() <= 1.0):
                raise ValueError(f"Mean probability for {task} out of bounds [0, 1].")
            if var.min().item() < 0.0:
                raise ValueError(f"Predictive variance for {task} is negative: {var.min().item()}.")
            if std.min().item() < 0.0:
                raise ValueError(f"Standard deviation for {task} is negative: {std.min().item()}.")
            if entropy.min().item() < 0.0:
                raise ValueError(f"Entropy for {task} is negative: {entropy.min().item()}.")
            if not (0.0 <= conf.min().item() and conf.max().item() <= 1.0):
                raise ValueError(f"Confidence score for {task} out of bounds [0, 1].")
            if not (0.0 <= conf_pct.min().item() and conf_pct.max().item() <= 100.0):
                raise ValueError(f"Confidence percentage for {task} out of bounds [0, 100].")

            if torch.isnan(p_mean).any() or torch.isnan(var).any() or torch.isnan(conf).any():
                raise ValueError(f"NaN value detected in {task} uncertainty statistics.")
            if torch.isinf(p_mean).any() or torch.isinf(var).any() or torch.isinf(conf).any():
                raise ValueError(f"Infinite value detected in {task} uncertainty statistics.")
