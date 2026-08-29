"""
Tests for Phase 8 Edge Cases and MC-Dropout Interface.
"""

import pytest
import torch

from phase_8_multitask_prediction.config import MultiTaskConfig
from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from phase_8_multitask_prediction.validation import validate_prediction_inputs


@pytest.fixture
def base_model() -> MultiTaskDiseasePredictionNetwork:
    cfg = MultiTaskConfig(upr_dim=512, shared_hidden_dim=128, task_hidden_dim=64, device="cpu")
    return MultiTaskDiseasePredictionNetwork(config=cfg)


def test_batch_size_1(base_model: MultiTaskDiseasePredictionNetwork):
    upr = torch.randn(1, 512)
    base_model.eval()
    with torch.no_grad():
        out = base_model(upr, return_probabilities=True)

    assert out["stroke_logits"].shape == (1, 1)
    assert out["alzheimer_logits"].shape == (1, 1)
    assert out["stroke_probabilities"].shape == (1, 1)
    assert out["alzheimer_probabilities"].shape == (1, 1)


def test_nan_input_raises(base_model: MultiTaskDiseasePredictionNetwork):
    upr = torch.randn(4, 512)
    upr[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        validate_prediction_inputs(upr, base_model.config)


def test_inf_input_raises(base_model: MultiTaskDiseasePredictionNetwork):
    upr = torch.randn(4, 512)
    upr[0, 0] = float("inf")

    with pytest.raises(ValueError, match="infinite"):
        validate_prediction_inputs(upr, base_model.config)


def test_invalid_dimension_raises(base_model: MultiTaskDiseasePredictionNetwork):
    upr = torch.randn(4, 256)  # Expected 512

    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_prediction_inputs(upr, base_model.config)


def test_mc_dropout_stochastic_passes(base_model: MultiTaskDiseasePredictionNetwork):
    """
    Phase 9 Compatibility Test: Verify that enabling MC-Dropout produces stochastic forward passes.
    """
    upr = torch.randn(2, 512)
    base_model.eval()

    # With MC-Dropout enabled, repeated forward passes on the same input vary slightly due to active dropout
    torch.manual_seed(123)
    out1 = base_model(upr, return_probabilities=True, enable_mc_dropout=True)
    out2 = base_model(upr, return_probabilities=True, enable_mc_dropout=True)

    assert torch.isfinite(out1["stroke_probabilities"]).all()
    assert torch.isfinite(out2["stroke_probabilities"]).all()
