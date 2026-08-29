"""
Tests for Unified Retinal Representation (URR) Head.
"""

import pytest
import torch

from phase_5_retinal_fusion.urr import (
    AttentionPoolingHead,
    UnifiedRetinalRepresentationHead,
)


def test_attention_pooling_head():
    head = AttentionPoolingHead(embed_dim=512, hidden_dim=256)
    tokens = torch.randn(4, 99, 512)
    pooled = head(tokens)
    assert pooled.shape == (4, 512)
    assert torch.isfinite(pooled).all()


def test_urr_head_output_shape():
    urr_head = UnifiedRetinalRepresentationHead(
        embed_dim=512,
        urr_dim=512,
        pooling_type="attention",
    )
    tokens = torch.randn(4, 99, 512)
    urr, urr_tokens = urr_head(tokens)

    assert urr.shape == (4, 512)
    assert urr_tokens.shape == (4, 99, 512)
    assert torch.isfinite(urr).all()
    assert torch.isfinite(urr_tokens).all()
