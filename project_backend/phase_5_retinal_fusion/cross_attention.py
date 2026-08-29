"""
Cross-Attention Fusion Module.

Implements multi-head cross-modal Transformer attention blocks allowing OCT-A, OCT-B,
and Fundus token streams to dynamically query, exchange context, and fuse information.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class MultiHeadCrossAttentionBlock(nn.Module):
    """
    Standard Transformer Multi-Head Cross-Attention Layer with Residual Connections,
    Pre-LayerNorm, and Feed-Forward Network (FFN).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-LayerNorms
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Multihead Attention (batch_first=True for [B, N, D])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

        self.drop_attn = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass: Query attends to Key/Value context.

        Args:
            query: Tensor [B, N_q, embed_dim]
            key_value: Context tensor [B, N_kv, embed_dim]
            key_padding_mask: Optional boolean mask [B, N_kv] where True indicates padding.

        Returns:
            Updated query representation [B, N_q, embed_dim].
        """
        # Pre-LayerNorm
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        # Cross-Attention
        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual connection
        x = query + self.drop_attn(attn_out)

        # FFN with residual
        x = x + self.ffn(self.norm_ffn(x))
        return x


class RetinalCrossAttentionFusion(nn.Module):
    """
    Multimodal Transformer Cross-Attention Engine.

    Aggregates reliability-modulated tokens from all available retinal modalities
    into a joint sequence, adds learnable modality-type embeddings, and processes them
    through a stack of self/cross-attention Transformer fusion layers.
    """

    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modalities = modalities or ["octa", "octb", "fundus"]
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Learnable modality type embeddings to preserve modality identity
        self.modality_embeddings = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for mod in self.modalities
        })

        # Stack of Transformer fusion layers
        self.layers = nn.ModuleList([
            MultiHeadCrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        modulated_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute cross-attention fusion over available modulated modality tokens.

        Args:
            modulated_features: Dict mapping modality name to tokens [B, N_m, embed_dim].

        Returns:
            Fused multimodal token sequence [B, N_total, embed_dim].
        """
        token_streams = []

        for mod in self.modalities:
            if mod in modulated_features:
                tokens = modulated_features[mod]  # [B, N_m, D]
                # Add modality-type embedding
                tokens_with_mod = tokens + self.modality_embeddings[mod]
                token_streams.append(tokens_with_mod)

        if not token_streams:
            raise ValueError("No modality tokens provided for cross-attention fusion.")

        # Concatenate along token sequence dimension -> [B, N_total, D]
        fused_sequence = torch.cat(token_streams, dim=1)

        # Pass through stacked Transformer fusion blocks (self/cross attention over joint sequence)
        for layer in self.layers:
            fused_sequence = layer(query=fused_sequence, key_value=fused_sequence)

        return self.final_norm(fused_sequence)
