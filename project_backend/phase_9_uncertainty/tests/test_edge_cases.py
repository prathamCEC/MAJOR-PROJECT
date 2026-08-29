"""
Tests for Phase 9 Edge Cases, Dimension Checks, and Error Handling.
"""

import pytest
import torch
import torch.nn as nn

from phase_9_uncertainty.config import UncertaintyConfig
from phase_9_uncertainty.engine import MCDropoutUncertaintyEngine
from phase_9_uncertainty.validation import validate_uncertainty_inputs
from phase_9_uncertainty.mc_dropout import run_mc_forward_passes


def test_batch_size_1():
    engine = MCDropoutUncertaintyEngine()
    upr = torch.randn(1, 512)

    res = engine.estimate_uncertainty(upr=upr, mc_samples=5)

    assert res["stroke"]["mc_mean_probability"].shape == (1,)
    assert res["alzheimer"]["mc_mean_probability"].shape == (1,)
    assert res["stroke"]["confidence_percent"].shape == (1,)


def test_large_batch_size():
    engine = MCDropoutUncertaintyEngine()
    upr = torch.randn(16, 512)

    res = engine.estimate_uncertainty(upr=upr, mc_samples=5)

    assert res["stroke"]["mc_mean_probability"].shape == (16,)
    assert res["alzheimer"]["mc_mean_probability"].shape == (16,)


def test_mc_samples_less_than_2_raises():
    cfg = UncertaintyConfig(mc_samples=1)
    with pytest.raises(ValueError, match="at least 2"):
        cfg.validate()

    engine = MCDropoutUncertaintyEngine()
    upr = torch.randn(2, 512)
    with pytest.raises(ValueError, match="at least 2"):
        engine.estimate_uncertainty(upr=upr, mc_samples=1)


def test_nan_input_raises():
    cfg = UncertaintyConfig()
    upr = torch.randn(2, 512)
    upr[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        validate_uncertainty_inputs(upr, cfg, mc_samples=5)


def test_inf_input_raises():
    cfg = UncertaintyConfig()
    upr = torch.randn(2, 512)
    upr[0, 0] = float("inf")

    with pytest.raises(ValueError, match="infinite"):
        validate_uncertainty_inputs(upr, cfg, mc_samples=5)


def test_no_dropout_model_raises_runtime_error():
    # Construct model without any dropout layers
    no_dropout_model = nn.Sequential(
        nn.Linear(512, 128),
        nn.GELU(),
        nn.Linear(128, 1),
    )
    upr = torch.randn(2, 512)

    with pytest.raises(RuntimeError, match="at least one active dropout layer"):
        run_mc_forward_passes(model=no_dropout_model, upr=upr, mc_samples=5)
