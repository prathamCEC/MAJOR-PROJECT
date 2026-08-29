"""
Validation and Numerical Safety Module for Phase 8 Multi-Task Prediction.

Performs input shape validation, NaN/Inf checks, label range verification, and output audits.
"""

from typing import Dict, Optional, Union
import torch

from .config import MultiTaskConfig


def validate_prediction_inputs(
    upr: torch.Tensor,
    config: MultiTaskConfig,
    stroke_targets: Optional[torch.Tensor] = None,
    alzheimer_targets: Optional[torch.Tensor] = None,
    stroke_mask: Optional[torch.Tensor] = None,
    alzheimer_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Validate inputs prior to multi-task disease classification.
    """
    if not isinstance(upr, torch.Tensor):
        raise TypeError(f"UPR input must be a torch.Tensor, got {type(upr)}.")

    # Rank & Dimension checks
    if upr.ndim not in (2, 3):
        raise ValueError(f"Expected UPR tensor of rank 2 ([B, D]) or rank 3 ([B, 1, D]), got rank {upr.ndim}.")

    if upr.shape[-1] != config.upr_dim:
        raise ValueError(
            f"Unified Patient Representation dimension mismatch: expected {config.upr_dim}, "
            f"but received {upr.shape[-1]}."
        )

    # NaN / Inf checks
    if torch.isnan(upr).any():
        raise ValueError("Unified Patient Representation contains NaN values.")
    if torch.isinf(upr).any():
        raise ValueError("Unified Patient Representation contains infinite (Inf) values.")

    batch_size = upr.shape[0]

    # Target & Mask checks
    if stroke_targets is not None:
        if stroke_targets.shape[0] != batch_size:
            raise ValueError(
                f"Stroke labels batch size mismatch: expected {batch_size}, got {stroke_targets.shape[0]}."
            )
    if alzheimer_targets is not None:
        if alzheimer_targets.shape[0] != batch_size:
            raise ValueError(
                f"Alzheimer's labels batch size mismatch: expected {batch_size}, got {alzheimer_targets.shape[0]}."
            )

    if stroke_mask is not None and stroke_mask.shape[0] != batch_size:
        raise ValueError(f"Stroke mask batch size mismatch: expected {batch_size}, got {stroke_mask.shape[0]}.")
    if alzheimer_mask is not None and alzheimer_mask.shape[0] != batch_size:
        raise ValueError(f"Alzheimer's mask batch size mismatch: expected {batch_size}, got {alzheimer_mask.shape[0]}.")


def validate_prediction_outputs(
    outputs: Dict[str, torch.Tensor],
    expected_batch_size: int,
) -> None:
    """
    Validate model output tensors.
    """
    for key in ("stroke_logits", "alzheimer_logits"):
        if key not in outputs:
            raise KeyError(f"Missing required prediction output key: '{key}'.")

        val = outputs[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"Output '{key}' must be a torch.Tensor, got {type(val)}.")

        if val.shape != (expected_batch_size, 1):
            raise ValueError(
                f"Output '{key}' shape mismatch: expected ({expected_batch_size}, 1), got {tuple(val.shape)}."
            )

        if torch.isnan(val).any():
            raise ValueError(f"Output '{key}' contains NaN values.")
        if torch.isinf(val).any():
            raise ValueError(f"Output '{key}' contains infinite (Inf) values.")
