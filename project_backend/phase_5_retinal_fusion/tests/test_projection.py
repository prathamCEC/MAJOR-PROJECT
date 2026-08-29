"""
Tests for modality feature projection module.
"""

import pytest
import torch

from phase_5_retinal_fusion.modality_projection import SingleModalityProjection, MultiModalityProjection


def test_single_modality_projection_shapes():
    proj = SingleModalityProjection(in_dim=768, embed_dim=512)

    # 1. 2D Pooled input [B, D]
    x_2d = torch.randn(4, 768)
    out_2d = proj(x_2d)
    assert out_2d.shape == (4, 1, 512)

    # 2. 3D Token input [B, N, D]
    x_3d = torch.randn(4, 49, 768)
    out_3d = proj(x_3d)
    assert out_3d.shape == (4, 49, 512)

    # 3. 4D Spatial input [B, H, W, D]
    x_4d = torch.randn(4, 7, 7, 768)
    out_4d = proj(x_4d)
    assert out_4d.shape == (4, 49, 512)


def test_projection_dimension_mismatch_raises():
    proj = SingleModalityProjection(in_dim=768, embed_dim=512)
    x_wrong = torch.randn(2, 512)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        proj(x_wrong)


def test_multi_modality_projection():
    multi_proj = MultiModalityProjection(
        input_dims={"octa": 768, "octb": 768, "fundus": 768},
        embed_dim=512,
    )
    feats = {
        "octa": torch.randn(2, 49, 768),
        "octb": torch.randn(2, 49, 768),
        "fundus": torch.randn(2, 768),
    }
    projected = multi_proj(feats)

    assert "octa" in projected and projected["octa"].shape == (2, 49, 512)
    assert "octb" in projected and projected["octb"].shape == (2, 49, 512)
    assert "fundus" in projected and projected["fundus"].shape == (2, 1, 512)
