"""
Tests for fine-grained Monte Carlo Dropout activation.

Verifies that enable_mc_dropout selectively switches ONLY dropout modules to training mode,
while keeping Linear and LayerNorm modules strictly in evaluation mode.
"""

import pytest
import torch
import torch.nn as nn

from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from phase_9_uncertainty.mc_dropout import enable_mc_dropout, disable_mc_dropout, DROPOUT_MODULE_TYPES


def test_enable_mc_dropout_selectivity():
    model = MultiTaskDiseasePredictionNetwork()

    # Step 1: Put in eval mode
    model.eval()

    # Step 2: Enable MC Dropout
    active_count = enable_mc_dropout(model)
    assert active_count > 0, "No dropout layers were detected in Phase 8 model."

    # Step 3: Audit all submodules
    dropout_found = 0
    non_dropout_found = 0

    for name, module in model.named_modules():
        if isinstance(module, DROPOUT_MODULE_TYPES):
            assert module.training is True, f"Dropout module '{name}' was not set to training mode."
            dropout_found += 1
        elif isinstance(module, (nn.Linear, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            assert module.training is False, f"Non-dropout module '{name}' was incorrectly set to training mode."
            non_dropout_found += 1

    assert dropout_found == active_count
    assert non_dropout_found > 0


def test_disable_mc_dropout_restores_eval():
    model = MultiTaskDiseasePredictionNetwork()
    enable_mc_dropout(model)
    disable_mc_dropout(model)

    # All modules should be in evaluation mode
    for name, module in model.named_modules():
        assert module.training is False, f"Module '{name}' was not restored to eval mode."
