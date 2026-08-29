"""
Unified Retinal Representation (URR) Head Module.

Aggregates fused multimodal token streams into a fixed-dimensional Unified Retinal
Representation (URR) vector for downstream clinical integration (Phase 6 / Phase 7).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPoolingHead(nn.Module):
    """
    Learned Attentive Pooling mechanism that summarizes a variable-length token sequence
    into a single summary vector.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Fused multimodal tokens [B, N, embed_dim]

        Returns:
            Pooled representation [B, embed_dim]
        """
        # [B, N, 1]
        attn_weights = F.softmax(self.attn_net(tokens), dim=1)
        # Weighted sum: [B, 1, N] @ [B, N, D] -> [B, 1, D] -> [B, D]
        pooled = torch.bmm(attn_weights.transpose(1, 2), tokens).squeeze(1)
        return pooled


class UnifiedRetinalRepresentationHead(nn.Module):
    """
    Final URR representation and projection block.

    Guarantees a fixed-dimensional output vector [B, urr_dim] irrespective of
    the number of active modalities (1, 2, or 3).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        urr_dim: int = 512,
        pooling_type: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.urr_dim = urr_dim
        self.pooling_type = pooling_type.lower()

        if self.pooling_type == "attention":
            self.pooler = AttentionPoolingHead(embed_dim=embed_dim)
        elif self.pooling_type == "mean":
            self.pooler = None
        else:
            self.pooler = AttentionPoolingHead(embed_dim=embed_dim)

        # Output projection and normalization
        self.proj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, urr_dim),
            nn.LayerNorm(urr_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(urr_dim, urr_dim),
            nn.LayerNorm(urr_dim),
        )

        # Token sequence projection
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, urr_dim),
            nn.LayerNorm(urr_dim),
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Unified Retinal Representation.

        Args:
            fused_tokens: Output from Cross-Attention fusion [B, N_total, embed_dim].

        Returns:
            Tuple of:
            - urr: Fixed-dimensional retinal vector [B, urr_dim]
            - urr_tokens: Fused token sequence [B, N_total, urr_dim]
        """
        if self.pooler is not None:
            pooled = self.pooler(fused_tokens)  # [B, embed_dim]
        else:
            pooled = torch.mean(fused_tokens, dim=1)  # [B, embed_dim]

        urr = self.proj_head(pooled)  # [B, urr_dim]
        urr_tokens = self.token_proj(fused_tokens)  # [B, N_total, urr_dim]

        return urr, urr_tokens
