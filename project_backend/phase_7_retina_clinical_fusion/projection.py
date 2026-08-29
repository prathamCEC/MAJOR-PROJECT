"""
Multimodal Projection Module for Phase 7 Retina-Clinical Fusion.

Transforms heterogeneous retinal (Phase 5) and clinical (Phase 6) representations into
a unified latent embedding space of dimension D_common with shape standardizations.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn


class RepresentationProjectionLayer(nn.Module):
    """
    Projects input representations from input_dim to common_embed_dim.
    Handles both single-vector [B, D] and token-sequence [B, N, D] representations.
    """

    def __init__(
        self,
        input_dim: int,
        common_embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.common_embed_dim = common_embed_dim

        if input_dim != common_embed_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, common_embed_dim),
                nn.LayerNorm(common_embed_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, D] or [B, N, D]

        Returns:
            Normalized token sequence [B, N, common_embed_dim] (N=1 if input was [B, D])
        """
        if x.ndim == 2:
            # Expand [B, D] -> [B, 1, D]
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected input tensor of rank 2 ([B, D]) or rank 3 ([B, N, D]), got rank {x.ndim}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected last dimension {self.input_dim}, "
                f"but received {x.shape[-1]}."
            )

        return self.proj(x)


class RetinaClinicalProjection(nn.Module):
    """
    Joint projection engine transforming both Retinal and Clinical streams.
    """

    def __init__(
        self,
        retinal_input_dim: int = 512,
        clinical_input_dim: int = 512,
        common_embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.retinal_proj = RepresentationProjectionLayer(
            input_dim=retinal_input_dim,
            common_embed_dim=common_embed_dim,
            dropout=dropout,
        )
        self.clinical_proj = RepresentationProjectionLayer(
            input_dim=clinical_input_dim,
            common_embed_dim=common_embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        retinal_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            retinal_features: [B, D_ret] or [B, N, D_ret]
            clinical_features: [B, D_clin] or [B, M, D_clin]

        Returns:
            Tuple of:
            - projected_retinal: [B, N, common_embed_dim]
            - projected_clinical: [B, M, common_embed_dim]
        """
        if retinal_features.shape[0] != clinical_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: retinal has batch size {retinal_features.shape[0]}, "
                f"while clinical has batch size {clinical_features.shape[0]}."
            )

        proj_ret = self.retinal_proj(retinal_features)
        proj_clin = self.clinical_proj(clinical_features)

        return proj_ret, proj_clin
