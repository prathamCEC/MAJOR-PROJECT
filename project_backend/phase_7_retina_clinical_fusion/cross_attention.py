"""
Bidirectional Retina-Clinical Cross-Attention Module.

Enables deep multimodal contextualization where retinal feature tokens query clinical history
and clinical tokens query retinal imaging microvascular patterns.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Directional Cross-Attention Block (Query from Stream A, Key/Value from Stream B).
    Pre-LayerNorm -> Multi-Head Cross Attention -> Residual -> Pre-LayerNorm -> FFN -> Residual.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-LayerNorms
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Multi-Head Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
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
        query_stream: torch.Tensor,
        kv_stream: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_stream: Queries [B, N_q, embed_dim]
            kv_stream: Keys & Values [B, N_kv, embed_dim]
            key_padding_mask: Optional mask for kv_stream [B, N_kv]

        Returns:
            Enhanced query_stream [B, N_q, embed_dim]
        """
        q_norm = self.norm_query(query_stream)
        kv_norm = self.norm_kv(kv_stream)

        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual connection 1
        x = query_stream + self.drop_attn(attn_out)

        # Residual connection 2 (FFN)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class BidirectionalRetinaClinicalBlock(nn.Module):
    """
    Simultaneous Bidirectional Cross-Attention:
    1. Retina attends to Clinical (Retina <- Clinical)
    2. Clinical attends to Retina (Clinical <- Retina)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        # Retina -> Clinical attention (Retina queries, Clinical keys/values)
        self.retina_cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # Clinical -> Retina attention (Clinical queries, Retina keys/values)
        self.clinical_cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

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
            retinal_mask: [B, N] (True = padding)
            clinical_mask: [B, M] (True = padding)

        Returns:
            Tuple of:
            - enhanced_retinal: [B, N, embed_dim]
            - enhanced_clinical: [B, M, embed_dim]
        """
        # Execute bidirectional cross-attention with cross-masks
        new_retinal = self.retina_cross_attn(
            query_stream=retinal_tokens,
            kv_stream=clinical_tokens,
            key_padding_mask=clinical_mask,
        )

        new_clinical = self.clinical_cross_attn(
            query_stream=clinical_tokens,
            kv_stream=retinal_tokens,
            key_padding_mask=retinal_mask,
        )

        return new_retinal, new_clinical


class BidirectionalRetinaClinicalTransformer(nn.Module):
    """
    Stacked Multi-Layer Bidirectional Cross-Attention Transformer.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            BidirectionalRetinaClinicalBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm_retinal = nn.LayerNorm(embed_dim)
        self.norm_clinical = nn.LayerNorm(embed_dim)

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
            Tuple of contextualized [B, N, embed_dim] and [B, M, embed_dim]
        """
        ret_curr = retinal_tokens
        clin_curr = clinical_tokens

        for layer in self.layers:
            ret_curr, clin_curr = layer(
                retinal_tokens=ret_curr,
                clinical_tokens=clin_curr,
                retinal_mask=retinal_mask,
                clinical_mask=clinical_mask,
            )

        return self.norm_retinal(ret_curr), self.norm_clinical(clin_curr)
