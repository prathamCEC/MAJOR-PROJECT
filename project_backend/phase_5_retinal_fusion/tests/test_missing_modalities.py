"""
Tests for missing modality handling across all possible input combinations.
"""

import pytest
import torch

from phase_5_retinal_fusion.config import FusionConfig
from phase_5_retinal_fusion.fusion_model import RetinalMultimodalFusionModel


@pytest.fixture
def fusion_model() -> RetinalMultimodalFusionModel:
    cfg = FusionConfig(embed_dim=256, urr_dim=512, device="cpu")
    return RetinalMultimodalFusionModel(config=cfg)


def test_missing_modality_cases(fusion_model: RetinalMultimodalFusionModel):
    B = 3
    full_feats = {
        "octa": torch.randn(B, 49, 768),
        "octb": torch.randn(B, 49, 768),
        "fundus": torch.randn(B, 768),
    }

    # Case 1: All 3 modalities present (OCTA + OCTB + Fundus)
    out1 = fusion_model(full_feats)
    assert out1["urr"].shape == (B, 512)
    assert len(out1["modality_weights"]) == 3

    # Case 2: 2 Modalities (OCTA + OCTB)
    feats_2a = {"octa": full_feats["octa"], "octb": full_feats["octb"]}
    out2a = fusion_model(feats_2a)
    assert out2a["urr"].shape == (B, 512)
    assert len(out2a["modality_weights"]) == 2

    # Case 3: 2 Modalities (OCTA + Fundus)
    feats_2b = {"octa": full_feats["octa"], "fundus": full_feats["fundus"]}
    out2b = fusion_model(feats_2b)
    assert out2b["urr"].shape == (B, 512)
    assert len(out2b["modality_weights"]) == 2

    # Case 4: 2 Modalities (OCTB + Fundus)
    feats_2c = {"octb": full_feats["octb"], "fundus": full_feats["fundus"]}
    out2c = fusion_model(feats_2c)
    assert out2c["urr"].shape == (B, 512)
    assert len(out2c["modality_weights"]) == 2

    # Case 5: Single Modality Only
    for single_mod in ["octa", "octb", "fundus"]:
        feats_single = {single_mod: full_feats[single_mod]}
        out_single = fusion_model(feats_single)
        assert out_single["urr"].shape == (B, 512)
        assert len(out_single["modality_weights"]) == 1
        # Single modality weight must be 1.0
        assert torch.allclose(out_single["modality_weights"][single_mod], torch.ones(B, 1), atol=1e-5)
