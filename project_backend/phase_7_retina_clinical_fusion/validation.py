"""
Numerical Safety and Validation Module for Phase 7 Retina-Clinical Fusion.

Performs shape consistency checks, batch size alignment, finite value validation,
and head divisibility audits to ensure robust multimodal computation.
"""

from typing import Dict, Optional, Tuple, Union
import torch

from .config import RetinaClinicalConfig


def validate_fusion_inputs(
    retinal_representation: torch.Tensor,
    clinical_representation: torch.Tensor,
    config: RetinaClinicalConfig,
    retinal_mask: Optional[torch.Tensor] = None,
    clinical_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Validate input tensors prior to cross-attention fusion.
    """
    # 1. Type validation
    if not isinstance(retinal_representation, torch.Tensor):
        raise TypeError(f"Retinal representation must be a torch.Tensor, got {type(retinal_representation)}.")
    if not isinstance(clinical_representation, torch.Tensor):
        raise TypeError(f"Clinical representation must be a torch.Tensor, got {type(clinical_representation)}.")

    # 2. Rank validation
    if retinal_representation.ndim not in (2, 3):
        raise ValueError(
            f"Retinal representation must have shape [B, D] or [B, N, D], "
            f"got rank {retinal_representation.ndim} (shape: {tuple(retinal_representation.shape)})."
        )
    if clinical_representation.ndim not in (2, 3):
        raise ValueError(
            f"Clinical representation must have shape [B, D] or [B, M, D], "
            f"got rank {clinical_representation.ndim} (shape: {tuple(clinical_representation.shape)})."
        )

    # 3. Batch size alignment
    batch_size_ret = retinal_representation.shape[0]
    batch_size_clin = clinical_representation.shape[0]
    if batch_size_ret != batch_size_clin:
        raise ValueError(
            f"Batch size mismatch: retinal batch size is {batch_size_ret}, "
            f"while clinical batch size is {batch_size_clin}."
        )

    # 4. Feature dimension validation
    ret_dim = retinal_representation.shape[-1]
    clin_dim = clinical_representation.shape[-1]
    if ret_dim != config.retinal_input_dim:
        raise ValueError(
            f"Retinal feature dimension mismatch: expected {config.retinal_input_dim}, "
            f"but received {ret_dim}."
        )
    if clin_dim != config.clinical_input_dim:
        raise ValueError(
            f"Clinical feature dimension mismatch: expected {config.clinical_input_dim}, "
            f"but received {clin_dim}."
        )

    # 5. Attention head divisibility
    if config.common_embed_dim % config.num_heads != 0:
        raise ValueError(
            f"Embedding dimension ({config.common_embed_dim}) must be divisible by "
            f"number of attention heads ({config.num_heads})."
        )

    # 6. NaN and Inf Audits
    if torch.isnan(retinal_representation).any():
        raise ValueError("Retinal representation contains NaN values.")
    if torch.isinf(retinal_representation).any():
        raise ValueError("Retinal representation contains infinite (Inf) values.")
    if torch.isnan(clinical_representation).any():
        raise ValueError("Clinical representation contains NaN values.")
    if torch.isinf(clinical_representation).any():
        raise ValueError("Clinical representation contains infinite (Inf) values.")

    # 7. Mask validations
    if retinal_mask is not None:
        if retinal_mask.shape[0] != batch_size_ret:
            raise ValueError(f"Retinal mask batch size mismatch: {retinal_mask.shape[0]} vs {batch_size_ret}.")
    if clinical_mask is not None:
        if clinical_mask.shape[0] != batch_size_clin:
            raise ValueError(f"Clinical mask batch size mismatch: {clinical_mask.shape[0]} vs {batch_size_clin}.")


def validate_upr_output(
    upr: torch.Tensor,
    expected_batch_size: int,
    expected_dim: int = 512,
) -> None:
    """
    Validate Unified Patient Representation output tensor.
    """
    if not isinstance(upr, torch.Tensor):
        raise TypeError(f"UPR output must be a torch.Tensor, got {type(upr)}.")

    if upr.shape != (expected_batch_size, expected_dim):
        raise ValueError(
            f"Expected Unified Patient Representation shape ({expected_batch_size}, {expected_dim}), "
            f"but received {tuple(upr.shape)}."
        )

    if torch.isnan(upr).any():
        raise ValueError("Unified Patient Representation contains NaN values.")

    if torch.isinf(upr).any():
        raise ValueError("Unified Patient Representation contains infinite (Inf) values.")
