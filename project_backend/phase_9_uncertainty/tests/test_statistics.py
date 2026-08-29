"""
Tests for uncertainty statistical aggregation and confidence scoring.
"""

import numpy as np
import pytest
import torch

from phase_9_uncertainty.uncertainty import calculate_predictive_statistics, calculate_predictive_entropy
from phase_9_uncertainty.confidence import calculate_confidence


def test_predictive_statistics_manual_validation():
    # 2 batch samples, 4 MC passes
    mc_probs = torch.tensor([
        [0.2, 0.4, 0.6, 0.8],  # Mean = 0.5, Var = 0.066667
        [0.9, 0.9, 0.9, 0.9],  # Mean = 0.9, Var = 0.0
    ], dtype=torch.float32)

    stats = calculate_predictive_statistics(mc_probs)

    assert torch.isclose(stats["mean_probability"][0], torch.tensor(0.5))
    assert torch.isclose(stats["mean_probability"][1], torch.tensor(0.9))

    assert stats["variance"][0].item() > 0.0
    assert torch.isclose(stats["variance"][1], torch.tensor(0.0), atol=1e-6)

    assert torch.isclose(stats["std_deviation"][1], torch.tensor(0.0), atol=1e-6)
    assert stats["std_deviation"][0].item() > 0.0

    # Entropy should be maximum at p=0.5 (ln(2) ~= 0.6931)
    assert np.isclose(stats["entropy"][0].item(), np.log(2.0), atol=1e-3)
    assert stats["entropy"][1].item() < stats["entropy"][0].item()


def test_confidence_calculation_bounds():
    variances = torch.tensor([0.0, 0.125, 0.25, 0.5])  # 0.5 exceeds scale 0.25 -> clamped to 1.0

    conf_res = calculate_confidence(variances, uncertainty_scale=0.25)
    conf = conf_res["confidence"]
    pct = conf_res["confidence_percent"]

    assert torch.isclose(conf[0], torch.tensor(1.0))
    assert torch.isclose(pct[0], torch.tensor(100.0))

    assert torch.isclose(conf[1], torch.tensor(0.5))
    assert torch.isclose(pct[1], torch.tensor(50.0))

    assert torch.isclose(conf[2], torch.tensor(0.0))
    assert torch.isclose(pct[2], torch.tensor(0.0))

    assert torch.isclose(conf[3], torch.tensor(0.0))
    assert torch.isclose(pct[3], torch.tensor(0.0))
