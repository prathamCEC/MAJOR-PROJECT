"""
Tests for Transformer Cross-Attention module.
"""

import pytest
import torch

from phase_5_retinal_fusion.cross_attention import (
    MultiHeadCrossAttentionBlock,
    RetinalCrossAttentionFusion,
)


def test_multihead_cross_attention_block():
    block = MultiHeadCrossAttentionBlock(embed_dim=512, num_heads=8, ffn_dim=1024)
    q = torch.randn(2, 10, 512)
    kv = torch.randn(2, 20, 512)

    out = block(query=q, key_value=kv)
    assert out.shape == (2, 10, 512)
    assert torch.isfinite(out).all()


def test_retinal_cross_attention_fusion():
    fusion = RetinalCrossAttentionFusion(
        modalities=["octa", "octb", "fundus"],
        embed_dim=512,
        num_heads=8,
        num_layers=2,
    )
    modulated = {
        "octa": torch.randn(2, 49, 512),
        "octb": torch.randn(2, 49, 512),
        "fundus": torch.randn(2, 1, 512),
    }

    # Total tokens = 49 + 49 + 1 = 99
    fused = fusion(modulated)
    assert fused.shape == (2, 99, 512)
    assert torch.isfinite(fused).all()
