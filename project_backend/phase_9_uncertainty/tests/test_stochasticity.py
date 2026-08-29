"""
Tests for Monte Carlo Dropout stochasticity and forward sampling diversity.
"""

import pytest
import torch

from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from phase_9_uncertainty.mc_dropout import run_mc_forward_passes


def test_mc_forward_passes_generate_stochastic_variance():
    model = MultiTaskDiseasePredictionNetwork()
    B = 2
    T = 15
    upr = torch.randn(B, 512)

    res = run_mc_forward_passes(model=model, upr=upr, mc_samples=T)

    assert "stroke_probabilities" in res
    assert "alzheimer_probabilities" in res
    assert res["stroke_probabilities"].shape == (B, T)
    assert res["alzheimer_probabilities"].shape == (B, T)

    # Check that across T samples, predictions are not all identical (variance > 0)
    stroke_var = torch.var(res["stroke_probabilities"], dim=1)
    alz_var = torch.var(res["alzheimer_probabilities"], dim=1)

    assert (stroke_var >= 0.0).all()
    assert (alz_var >= 0.0).all()
    # At least one sample should exhibit non-zero variance due to active dropout
    assert stroke_var.sum().item() > 0.0 or alz_var.sum().item() > 0.0


def test_mc_forward_passes_bounds():
    model = MultiTaskDiseasePredictionNetwork()
    B = 4
    T = 10
    upr = torch.randn(B, 512)

    res = run_mc_forward_passes(model=model, upr=upr, mc_samples=T)

    st_p = res["stroke_probabilities"]
    al_p = res["alzheimer_probabilities"]

    assert (st_p >= 0.0).all() and (st_p <= 1.0).all()
    assert (al_p >= 0.0).all() and (al_p <= 1.0).all()
    assert torch.isfinite(st_p).all()
    assert torch.isfinite(al_p).all()
