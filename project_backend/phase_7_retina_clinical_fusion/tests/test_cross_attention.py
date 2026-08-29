"""
Tests for Bidirectional Cross-Attention Transformer.
"""

import pytest
import torch

from phase_7_retina_clinical_fusion.cross_attention import (
    CrossAttentionBlock,
    BidirectionalRetinaClinicalBlock,
    BidirectionalRetinaClinicalTransformer,
)


def test_cross_attention_block():
    block = CrossAttentionBlock(embed_dim=256, num_heads=4, ffn_dim=512)
    q = torch.randn(2, 3, 256)
    kv = torch.randn(2, 5, 256)

    out = block(query_stream=q, kv_stream=kv)
    assert out.shape == (2, 3, 256)
    assert torch.isfinite(out).all()


def test_bidirectional_block():
    block = BidirectionalRetinaClinicalBlock(embed_dim=256, num_heads=4, ffn_dim=512)
    ret = torch.randn(3, 4, 256)
    clin = torch.randn(3, 6, 256)

    new_ret, new_clin = block(retinal_tokens=ret, clinical_tokens=clin)
    assert new_ret.shape == (3, 4, 256)
    assert new_clin.shape == (3, 6, 256)
    assert torch.isfinite(new_ret).all()
    assert torch.isfinite(new_clin).all()


def test_bidirectional_transformer_stack():
    transformer = BidirectionalRetinaClinicalTransformer(
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
    )
    ret = torch.randn(2, 1, 256)
    clin = torch.randn(2, 8, 256)

    enh_ret, enh_clin = transformer(retinal_tokens=ret, clinical_tokens=clin)
    assert enh_ret.shape == (2, 1, 256)
    assert enh_clin.shape == (2, 8, 256)
    assert torch.isfinite(enh_ret).all()
    assert torch.isfinite(enh_clin).all()
