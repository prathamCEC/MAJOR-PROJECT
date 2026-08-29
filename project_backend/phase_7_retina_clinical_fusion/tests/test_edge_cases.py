"""
Tests for Phase 7 Edge Cases and Numerical Robustness.
"""

import pytest
import torch

from phase_7_retina_clinical_fusion.config import RetinaClinicalConfig
from phase_7_retina_clinical_fusion.fusion_model import RetinaClinicalFusionModel


@pytest.fixture
def base_model() -> RetinaClinicalFusionModel:
    cfg = RetinaClinicalConfig(
        retinal_input_dim=512,
        clinical_input_dim=512,
        common_embed_dim=256,
        num_heads=4,
        num_layers=2,
        upr_dim=512,
        device="cpu",
    )
    return RetinaClinicalFusionModel(config=cfg)


def test_edge_case_shapes(base_model: RetinaClinicalFusionModel):
    B = 3

    # Case 1: [B, D] + [B, D]
    out1 = base_model(torch.randn(B, 512), torch.randn(B, 512))
    assert out1["upr"].shape == (B, 512)

    # Case 2: [B, N, D] + [B, 1, D]
    out2 = base_model(torch.randn(B, 4, 512), torch.randn(B, 1, 512))
    assert out2["upr"].shape == (B, 512)

    # Case 3: [B, N, D] + [B, M, D]
    out3 = base_model(torch.randn(B, 3, 512), torch.randn(B, 6, 512))
    assert out3["upr"].shape == (B, 512)


def test_edge_case_batch_mismatch_raises(base_model: RetinaClinicalFusionModel):
    with pytest.raises(ValueError, match="Batch size mismatch"):
        base_model(torch.randn(4, 512), torch.randn(2, 512))


def test_edge_case_dimension_mismatch_raises(base_model: RetinaClinicalFusionModel):
    with pytest.raises(ValueError, match="Retinal feature dimension mismatch"):
        base_model(torch.randn(2, 256), torch.randn(2, 512))  # Retinal dim is 256 instead of 512


def test_edge_case_nan_input_raises(base_model: RetinaClinicalFusionModel):
    nan_tensor = torch.randn(2, 512)
    nan_tensor[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        base_model(nan_tensor, torch.randn(2, 512))


def test_edge_case_inf_input_raises(base_model: RetinaClinicalFusionModel):
    inf_tensor = torch.randn(2, 512)
    inf_tensor[0, 0] = float("inf")
    with pytest.raises(ValueError, match="infinite"):
        base_model(torch.randn(2, 512), inf_tensor)


def test_edge_case_train_eval_modes(base_model: RetinaClinicalFusionModel):
    ret = torch.randn(2, 512)
    clin = torch.randn(2, 512)

    # Train mode
    base_model.train()
    out_train = base_model(ret, clin)
    assert torch.isfinite(out_train["upr"]).all()

    # Eval mode (deterministic)
    base_model.eval()
    with torch.no_grad():
        out_eval1 = base_model(ret, clin)
        out_eval2 = base_model(ret, clin)
    assert torch.allclose(out_eval1["upr"], out_eval2["upr"])
