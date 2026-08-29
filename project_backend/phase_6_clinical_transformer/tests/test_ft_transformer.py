"""
Tests for FT-Transformer blocks and backbone.
"""

import pytest
import torch

from phase_6_clinical_transformer.ft_transformer import (
    FTTransformerBlock,
    FTTransformerBackbone,
)


def test_ft_transformer_block():
    block = FTTransformerBlock(embed_dim=256, num_heads=8, ffn_dim=512)
    x = torch.randn(4, 10, 256)
    out = block(x)
    assert out.shape == (4, 10, 256)
    assert torch.isfinite(out).all()


def test_ft_transformer_backbone():
    backbone = FTTransformerBackbone(
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        ffn_dim=512,
    )
    tokens = torch.randn(4, 12, 256)
    out = backbone(tokens)
    assert out.shape == (4, 12, 256)
    assert torch.isfinite(out).all()
