"""
FT-Transformer Backbone Module.

Implements standard Feature Tokenizer Transformer (FT-Transformer) blocks with
multi-head self-attention, Pre-LayerNorm, residual connections, and feed-forward networks.
"""

from typing import Optional
import torch
import torch.nn as nn


class FTTransformerBlock(nn.Module):
    """
    Standard FT-Transformer Layer Block.

    Pre-LayerNorm -> Multi-Head Self-Attention -> Residual -> Pre-LayerNorm -> FFN -> Residual.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-LayerNorms
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

        self.drop_attn = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Token tensor [B, 1 + N_feat, embed_dim]
            key_padding_mask: Optional boolean mask [B, 1 + N_feat]

        Returns:
            Updated token tensor [B, 1 + N_feat, embed_dim]
        """
        # 1. Self-Attention with Pre-LN and Residual
        x_norm = self.norm_attn(x)
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_attn(attn_out)

        # 2. FFN with Pre-LN and Residual
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FTTransformerBackbone(nn.Module):
    """
    Stacked Transformer Blocks for Clinical Feature Representation.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.blocks = nn.ModuleList([
            FTTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: Input token sequence [B, 1 + N_feat, embed_dim]

        Returns:
            Processed token sequence [B, 1 + N_feat, embed_dim]
        """
        x = tokens
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return self.final_norm(x)
