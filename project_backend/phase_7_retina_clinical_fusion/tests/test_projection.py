"""
Tests for Phase 7 Multimodal Projections.
"""

import pytest
import torch

from phase_7_retina_clinical_fusion.projection import (
    RepresentationProjectionLayer,
    RetinaClinicalProjection,
)


def test_projection_layer_rank2_and_rank3():
    proj = RepresentationProjectionLayer(input_dim=512, common_embed_dim=256)

    # Rank 2 input: [B, D] -> [B, 1, D_common]
    x_rank2 = torch.randn(4, 512)
    out_rank2 = proj(x_rank2)
    assert out_rank2.shape == (4, 1, 256)
    assert torch.isfinite(out_rank2).all()

    # Rank 3 input: [B, N, D] -> [B, N, D_common]
    x_rank3 = torch.randn(4, 5, 512)
    out_rank3 = proj(x_rank3)
    assert out_rank3.shape == (4, 5, 256)
    assert torch.isfinite(out_rank3).all()


def test_joint_retina_clinical_projection():
    joint_proj = RetinaClinicalProjection(
        retinal_input_dim=512,
        clinical_input_dim=256,
        common_embed_dim=512,
    )
    retinal = torch.randn(3, 512)
    clinical = torch.randn(3, 10, 256)

    proj_ret, proj_clin = joint_proj(retinal, clinical)
    assert proj_ret.shape == (3, 1, 512)
    assert proj_clin.shape == (3, 10, 512)


def test_batch_mismatch_raises():
    joint_proj = RetinaClinicalProjection(512, 512, 512)
    retinal = torch.randn(3, 512)
    clinical = torch.randn(5, 512)  # Mismatched batch size

    with pytest.raises(ValueError, match="Batch size mismatch"):
        joint_proj(retinal, clinical)
