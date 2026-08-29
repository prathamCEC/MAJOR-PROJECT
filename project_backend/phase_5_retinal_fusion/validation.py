"""
Validation Utilities for Phase 5 Retinal Fusion.

Provides tensor shape auditing, finite number checks (NaN/Inf), modality mask
validation, and dimension integrity verification.
"""

from typing import Dict, List, Optional
import torch


def validate_input_features(
    modality_features: Dict[str, torch.Tensor],
    expected_input_dims: Optional[Dict[str, int]] = None,
) -> None:
    """
    Validate incoming modality feature tensors before fusion.

    Args:
        modality_features: Dict mapping modality name to tensor.
        expected_input_dims: Dict mapping modality name to expected feature dimension.

    Raises:
        ValueError: If tensors are invalid, empty, containing NaN/Inf, or dimension mismatch.
    """
    if not modality_features:
        raise ValueError("Modality features dictionary is empty. At least one modality is required.")

    expected_dims = expected_input_dims or {"octa": 768, "octb": 768, "fundus": 768}
    batch_sizes = []

    for mod, feat in modality_features.items():
        if not isinstance(feat, torch.Tensor):
            raise TypeError(f"Feature for modality '{mod}' must be a torch.Tensor, got {type(feat)}.")

        if feat.numel() == 0:
            raise ValueError(f"Feature tensor for modality '{mod}' is empty.")

        # Finite checks
        if torch.isnan(feat).any():
            raise ValueError(f"Feature tensor for modality '{mod}' contains NaN values.")

        if torch.isinf(feat).any():
            raise ValueError(f"Feature tensor for modality '{mod}' contains Inf values.")

        # Dimension checks
        if feat.ndim not in (2, 3, 4):
            raise ValueError(
                f"Modality '{mod}' tensor must have 2, 3, or 4 dimensions [B, D], [B, N, D], or [B, H, W, D], "
                f"but received shape {tuple(feat.shape)}."
            )

        batch_sizes.append(feat.shape[0])

        last_dim = feat.shape[-1]
        if mod in expected_dims and last_dim != expected_dims[mod]:
            raise ValueError(
                f"Expected feature dimension {expected_dims[mod]} for modality '{mod}', "
                f"but received {last_dim} (shape {tuple(feat.shape)})."
            )

    # Check consistent batch size
    if len(set(batch_sizes)) > 1:
        raise ValueError(
            f"Inconsistent batch sizes across modalities: {dict(zip(modality_features.keys(), batch_sizes))}."
        )


def validate_modality_mask(
    modality_mask: Dict[str, torch.Tensor],
    batch_size: int,
    modalities: Optional[List[str]] = None,
) -> None:
    """
    Validate modality mask tensors.

    Args:
        modality_mask: Dict mapping modality name to binary availability mask [B, 1].
        batch_size: Target batch size.
        modalities: List of expected modality names.
    """
    mods = modalities or ["octa", "octb", "fundus"]

    for mod in mods:
        if mod in modality_mask:
            mask_t = modality_mask[mod]
            if not isinstance(mask_t, torch.Tensor):
                raise TypeError(f"Mask for '{mod}' must be a torch.Tensor, got {type(mask_t)}.")

            if mask_t.shape[0] != batch_size:
                raise ValueError(
                    f"Mask batch size for '{mod}' ({mask_t.shape[0]}) does not match data batch size ({batch_size})."
                )

            # Binary check
            unique_vals = torch.unique(mask_t).cpu().tolist()
            for v in unique_vals:
                if v not in (0.0, 1.0, 0, 1):
                    raise ValueError(
                        f"Mask for '{mod}' must contain only binary values (0 or 1), found values: {unique_vals}."
                    )


def validate_urr_output(
    urr_output: Dict[str, torch.Tensor],
    expected_batch_size: int,
    expected_urr_dim: int = 512,
) -> None:
    """
    Validate the output dictionary from RetinalMultimodalFusionModel.
    """
    required_keys = ["urr", "urr_tokens", "modality_weights", "projected_features", "fused_tokens"]
    for k in required_keys:
        if k not in urr_output:
            raise KeyError(f"Missing expected output key in fusion result: '{k}'.")

    urr = urr_output["urr"]
    if urr.shape != (expected_batch_size, expected_urr_dim):
        raise ValueError(
            f"Expected URR shape ({expected_batch_size}, {expected_urr_dim}), but received {tuple(urr.shape)}."
        )

    if torch.isnan(urr).any() or torch.isinf(urr).any():
        raise ValueError("URR vector contains NaN or Inf values.")
