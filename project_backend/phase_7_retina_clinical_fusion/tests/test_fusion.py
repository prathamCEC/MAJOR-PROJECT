"""
Tests for Pooling and Gated Multimodal Fusion.
"""

import pytest
import torch

from phase_7_retina_clinical_fusion.pooling import MultimodalTokenPooler
from phase_7_retina_clinical_fusion.fusion import GatedMultimodalFusion


def test_multimodal_token_pooler():
    pooler = MultimodalTokenPooler(embed_dim=256, strategy="attentive")
    ret_tokens = torch.randn(4, 3, 256)
    clin_tokens = torch.randn(4, 5, 256)

    v_ret, v_clin = pooler(retinal_tokens=ret_tokens, clinical_tokens=clin_tokens)
    assert v_ret.shape == (4, 256)
    assert v_clin.shape == (4, 256)
    assert torch.isfinite(v_ret).all()
    assert torch.isfinite(v_clin).all()


def test_gated_multimodal_fusion():
    fusion = GatedMultimodalFusion(embed_dim=256, upr_dim=512)
    v_ret = torch.randn(4, 256)
    v_clin = torch.randn(4, 256)

    upr, fused_vec, gate_weights = fusion(v_ret, v_clin)

    assert upr.shape == (4, 512)
    assert fused_vec.shape == (4, 256)
    assert gate_weights.shape == (4, 256)
    assert (gate_weights >= 0.0).all() and (gate_weights <= 1.0).all()
    assert torch.isfinite(upr).all()
