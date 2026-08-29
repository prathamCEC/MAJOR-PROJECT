"""
Tests for Dynamic Modality Reliability Attention (DMRA).
"""

import pytest
import torch

from phase_5_retinal_fusion.reliability_attention import (
    SingleModalityReliabilityScorer,
    DynamicModalityReliabilityAttention,
)


def test_single_scorer():
    scorer = SingleModalityReliabilityScorer(embed_dim=512, hidden_dim=256)
    x = torch.randn(4, 49, 512)
    logit = scorer(x)
    assert logit.shape == (4, 1)
    assert torch.isfinite(logit).all()


def test_dmra_softmax_sums_to_one():
    dmra = DynamicModalityReliabilityAttention(modalities=["octa", "octb", "fundus"], embed_dim=512)
    feats = {
        "octa": torch.randn(3, 49, 512),
        "octb": torch.randn(3, 49, 512),
        "fundus": torch.randn(3, 1, 512),
    }

    modulated, weights, logits = dmra(feats)

    assert len(weights) == 3
    # Check that weights sum to 1.0 across modalities for each batch item
    total_w = sum(weights[m] for m in ["octa", "octb", "fundus"])
    assert torch.allclose(total_w, torch.ones(3, 1), atol=1e-5)

    # Check each modulated tensor shape matches original
    for m in ["octa", "octb", "fundus"]:
        assert modulated[m].shape == feats[m].shape
        assert torch.all(weights[m] >= 0.0)


def test_dmra_with_missing_modality_mask():
    dmra = DynamicModalityReliabilityAttention(modalities=["octa", "octb", "fundus"], embed_dim=512)
    feats = {
        "octa": torch.randn(2, 49, 512),
        "octb": torch.randn(2, 49, 512),
        "fundus": torch.randn(2, 1, 512),
    }
    # Fundus missing (mask = 0.0)
    mask = {
        "octa": torch.tensor([[1.0], [1.0]]),
        "octb": torch.tensor([[1.0], [1.0]]),
        "fundus": torch.tensor([[0.0], [0.0]]),
    }

    modulated, weights, _ = dmra(feats, modality_mask=mask)

    # Fundus weight must be 0.0
    assert torch.allclose(weights["fundus"], torch.zeros(2, 1), atol=1e-6)
    # OCTA + OCTB weights must sum to 1.0
    active_w = weights["octa"] + weights["octb"]
    assert torch.allclose(active_w, torch.ones(2, 1), atol=1e-5)
