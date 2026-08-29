"""
Shared Prediction Trunk Module for Phase 8 Multi-Task Network.

Transforms the Unified Patient Representation (UPR) into a shared multimodal disease latent
representation containing features common to both Stroke and Alzheimer's Disease prediction.
"""

import torch
import torch.nn as nn


class SharedPredictionTrunk(nn.Module):
    """
    Multi-layer shared representation processor for multimodal patient embeddings.
    """

    def __init__(
        self,
        upr_dim: int = 512,
        shared_hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.upr_dim = upr_dim
        self.shared_hidden_dim = shared_hidden_dim

        self.net = nn.Sequential(
            nn.LayerNorm(upr_dim),
            nn.Linear(upr_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, upr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            upr: Unified Patient Representation tensor [B, upr_dim]

        Returns:
            Shared representation tensor [B, shared_hidden_dim]
        """
        if upr.ndim == 3 and upr.shape[1] == 1:
            upr = upr.squeeze(1)

        if upr.ndim != 2:
            raise ValueError(f"Expected UPR tensor of rank 2 ([B, {self.upr_dim}]), got shape {tuple(upr.shape)}.")

        if upr.shape[-1] != self.upr_dim:
            raise ValueError(
                f"UPR feature dimension mismatch: expected {self.upr_dim}, but received {upr.shape[-1]}."
            )

        return self.net(upr)
