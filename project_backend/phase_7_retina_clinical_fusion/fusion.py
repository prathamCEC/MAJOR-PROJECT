"""
Multimodal Fusion and Unified Patient Representation (UPR) Module.

Combines enhanced retinal and clinical global vectors using a learned dynamic gating mechanism
and projects the result into the standardized Unified Patient Representation (UPR) space.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class GatedMultimodalFusion(nn.Module):
    """
    Learnable Gated Multimodal Fusion Mechanism.

    gate = sigmoid(Linear([v_retinal || v_clinical]))
    v_fused = gate * v_retinal + (1 - gate) * v_clinical
    """

    def __init__(
        self,
        embed_dim: int = 512,
        upr_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.upr_dim = upr_dim

        # Input Normalizations
        self.norm_ret = nn.LayerNorm(embed_dim)
        self.norm_clin = nn.LayerNorm(embed_dim)

        # Dynamic Gating Network: [B, 2 * embed_dim] -> [B, embed_dim]
        self.gate_net = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        # Final UPR Projection Network: [B, embed_dim] -> [B, upr_dim]
        self.upr_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, upr_dim),
            nn.LayerNorm(upr_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(upr_dim, upr_dim),
            nn.LayerNorm(upr_dim),
        )

    def forward(
        self,
        retinal_vector: torch.Tensor,
        clinical_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            retinal_vector: Pooled retinal representation [B, embed_dim]
            clinical_vector: Pooled clinical representation [B, embed_dim]

        Returns:
            Tuple of:
            - upr: Unified Patient Representation [B, upr_dim]
            - fused_vector: Pre-projection gated representation [B, embed_dim]
            - gate_weights: Learned multimodal gate activations [B, embed_dim]
        """
        norm_ret = self.norm_ret(retinal_vector)
        norm_clin = self.norm_clin(clinical_vector)

        # Concatenate normalized vectors [B, 2 * embed_dim]
        concat_feats = torch.cat([norm_ret, norm_clin], dim=-1)

        # Compute dynamic gate [B, embed_dim] with values in (0, 1)
        gate = self.gate_net(concat_feats)

        # Gated fusion with complementary balance
        fused = gate * norm_ret + (1.0 - gate) * norm_clin

        # Project into final Unified Patient Representation space
        upr = self.upr_projection(fused)

        return upr, fused, gate
