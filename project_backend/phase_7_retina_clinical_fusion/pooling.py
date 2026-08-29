"""
Token Pooling Module for Multimodal Sequence Aggregation.

Converts variable-length enhanced token sequences into fixed-dimensional global vectors.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveSequencePooler(nn.Module):
    """
    Learned attention-weighted pooling over token sequences.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, embed_dim]
            key_padding_mask: Optional [B, N] (True = padding)

        Returns:
            Pooled vector [B, embed_dim]
        """
        if tokens.shape[1] == 1:
            return tokens.squeeze(1)

        # Compute raw attention scores [B, N, 1]
        scores = self.net(tokens)

        if key_padding_mask is not None:
            # Mask out padding tokens (-inf before softmax)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(scores, dim=1)  # [B, N, 1]
        pooled = torch.bmm(weights.transpose(1, 2), tokens).squeeze(1)  # [B, embed_dim]
        return pooled


class MultimodalTokenPooler(nn.Module):
    """
    Pools enhanced Retinal and Clinical token streams into global feature vectors.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        strategy: str = "attentive",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.strategy = strategy.lower()

        if self.strategy == "attentive":
            self.retinal_pooler = AttentiveSequencePooler(embed_dim=embed_dim)
            self.clinical_pooler = AttentiveSequencePooler(embed_dim=embed_dim)
        else:
            self.retinal_pooler = None
            self.clinical_pooler = None

    def forward(
        self,
        retinal_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
        retinal_mask: Optional[torch.Tensor] = None,
        clinical_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            retinal_tokens: [B, N, embed_dim]
            clinical_tokens: [B, M, embed_dim]

        Returns:
            Tuple of [B, embed_dim] retinal_vector and [B, embed_dim] clinical_vector
        """
        if self.strategy == "attentive":
            v_ret = self.retinal_pooler(retinal_tokens, key_padding_mask=retinal_mask)
            v_clin = self.clinical_pooler(clinical_tokens, key_padding_mask=clinical_mask)
        elif self.strategy == "cls":
            v_ret = retinal_tokens[:, 0, :]
            v_clin = clinical_tokens[:, 0, :]
        elif self.strategy == "mean":
            v_ret = torch.mean(retinal_tokens, dim=1)
            v_clin = torch.mean(clinical_tokens, dim=1)
        else:
            v_ret = retinal_tokens[:, 0, :] if retinal_tokens.shape[1] == 1 else torch.mean(retinal_tokens, dim=1)
            v_clin = clinical_tokens[:, 0, :] if clinical_tokens.shape[1] == 1 else torch.mean(clinical_tokens, dim=1)

        return v_ret, v_clin
