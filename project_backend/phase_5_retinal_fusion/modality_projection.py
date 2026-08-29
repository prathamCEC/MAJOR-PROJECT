"""
Modality Feature Projection Module.

Maps modality-specific retinal representations (OCT-A, OCT-B, Fundus)
from their native Phase 4 feature dimensions into a common embedding space.
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn


class SingleModalityProjection(nn.Module):
    """
    Projects a single modality's feature representation into common fusion dimension.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection pass.

        Args:
            x: Input feature tensor of shape:
               - [B, in_dim] -> converted to [B, 1, embed_dim]
               - [B, N, in_dim] -> projected to [B, N, embed_dim]
               - [B, H, W, in_dim] -> flattened and projected to [B, H*W, embed_dim]

        Returns:
            Projected token tensor [B, N, embed_dim].
        """
        if x.ndim == 2:
            # [B, in_dim] -> [B, 1, in_dim]
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            # [B, H, W, in_dim] -> [B, H*W, in_dim]
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        elif x.ndim != 3:
            raise ValueError(
                f"Expected feature tensor with 2, 3, or 4 dimensions, got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected input dimension {self.in_dim}, "
                f"but received {x.shape[-1]} (shape {tuple(x.shape)})."
            )

        return self.proj(x)


class MultiModalityProjection(nn.Module):
    """
    Manages dedicated projection heads for all retinal modalities (OCT-A, OCT-B, Fundus).
    """

    def __init__(
        self,
        input_dims: Optional[Dict[str, int]] = None,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = input_dims or {"octa": 768, "octb": 768, "fundus": 768}
        self.embed_dim = embed_dim

        self.projections = nn.ModuleDict({
            mod: SingleModalityProjection(in_dim=dim, embed_dim=embed_dim, dropout=dropout)
            for mod, dim in dims.items()
        })

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project all available modality tensors.

        Args:
            modality_features: Dict mapping modality name ('octa', 'octb', 'fundus')
                               to feature tensors.

        Returns:
            Dict mapping modality name to projected tokens [B, N_m, embed_dim].
        """
        projected = {}
        for mod, feat in modality_features.items():
            if mod in self.projections:
                projected[mod] = self.projections[mod](feat)
            else:
                raise KeyError(
                    f"Unknown modality '{mod}'. Configured modalities: {list(self.projections.keys())}"
                )
        return projected
