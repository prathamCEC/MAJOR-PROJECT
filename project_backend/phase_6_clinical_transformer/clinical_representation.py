"""
Clinical Representation Head Module.

Extracts and projects the CLS representation from the FT-Transformer sequence into
a normalized, fixed-dimensional Clinical Representation (CR) vector ready for Phase 7.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveTokenPooler(nn.Module):
    """
    Learned attention pooling over feature tokens.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Feature tokens [B, N, embed_dim]
        Returns:
            Pooled representation [B, embed_dim]
        """
        attn = F.softmax(self.net(tokens), dim=1)  # [B, N, 1]
        pooled = torch.bmm(attn.transpose(1, 2), tokens).squeeze(1)  # [B, embed_dim]
        return pooled


class ClinicalRepresentationHead(nn.Module):
    """
    Produces the official fixed-dimensional Clinical Representation (CR) for Phase 7.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        clinical_representation_dim: int = 512,
        pooling_strategy: str = "cls",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clinical_representation_dim = clinical_representation_dim
        self.pooling_strategy = pooling_strategy.lower()

        if self.pooling_strategy == "attention":
            self.pooler = AttentiveTokenPooler(embed_dim=embed_dim)
        else:
            self.pooler = None

        # Multi-layer representation projection head
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, clinical_representation_dim),
            nn.LayerNorm(clinical_representation_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(clinical_representation_dim, clinical_representation_dim),
            nn.LayerNorm(clinical_representation_dim),
        )

    def forward(
        self,
        transformer_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            transformer_tokens: Processed sequence [B, 1 + N_feat, embed_dim]

        Returns:
            Tuple of:
            - clinical_representation: Output representation tensor [B, clinical_representation_dim]
            - pooled_cls: Raw CLS token representation [B, embed_dim]
        """
        if self.pooling_strategy == "cls":
            # Index 0 is the [CLS] token
            pooled_cls = transformer_tokens[:, 0, :]  # [B, embed_dim]
        elif self.pooling_strategy == "attention":
            # Exclude [CLS] and attentively pool feature tokens
            pooled_cls = self.pooler(transformer_tokens[:, 1:, :])
        elif self.pooling_strategy == "mean":
            # Average across all tokens
            pooled_cls = torch.mean(transformer_tokens, dim=1)
        else:
            pooled_cls = transformer_tokens[:, 0, :]

        # Project into final Clinical Representation space
        clinical_rep = self.proj(pooled_cls)  # [B, clinical_representation_dim]

        return clinical_rep, pooled_cls
