"""
Tests for MaskedMultiTaskLoss: missing label masking, task balancing, and zero-loss edge cases.
"""

import pytest
import torch

from phase_8_multitask_prediction.config import MultiTaskConfig
from phase_8_multitask_prediction.loss import MaskedMultiTaskLoss


def test_loss_both_tasks_available():
    loss_fn = MaskedMultiTaskLoss()
    B = 4
    st_logits = torch.randn(B, 1, requires_grad=True)
    al_logits = torch.randn(B, 1, requires_grad=True)
    st_targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    al_targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    losses = loss_fn(
        stroke_logits=st_logits,
        alzheimer_logits=al_logits,
        stroke_targets=st_targets,
        alzheimer_targets=al_targets,
    )

    assert "total_loss" in losses
    assert losses["total_loss"].item() > 0.0
    assert losses["stroke_valid_count"] == 4
    assert losses["alzheimer_valid_count"] == 4

    # Test backward pass
    losses["total_loss"].backward()
    assert st_logits.grad is not None
    assert al_logits.grad is not None


def test_loss_missing_stroke_labels():
    loss_fn = MaskedMultiTaskLoss()
    B = 4
    st_logits = torch.randn(B, 1, requires_grad=True)
    al_logits = torch.randn(B, 1, requires_grad=True)

    # Stroke has NO valid labels (all -1 or mask=0)
    st_targets = torch.tensor([[-1.0], [-1.0], [-1.0], [-1.0]])
    al_targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    losses = loss_fn(
        stroke_logits=st_logits,
        alzheimer_logits=al_logits,
        stroke_targets=st_targets,
        alzheimer_targets=al_targets,
    )

    assert losses["stroke_loss"].item() == 0.0
    assert losses["stroke_valid_count"] == 0
    assert losses["alzheimer_valid_count"] == 4
    assert torch.isclose(losses["total_loss"], losses["alzheimer_loss"])

    # Backward pass should propagate to Alzheimer's logits without NaN
    losses["total_loss"].backward()
    assert torch.isfinite(al_logits.grad).all()


def test_loss_all_labels_missing_safe_zero():
    loss_fn = MaskedMultiTaskLoss()
    B = 3
    st_logits = torch.randn(B, 1, requires_grad=True)
    al_logits = torch.randn(B, 1, requires_grad=True)

    losses = loss_fn(
        stroke_logits=st_logits,
        alzheimer_logits=al_logits,
        stroke_targets=None,
        alzheimer_targets=None,
    )

    assert losses["total_loss"].item() == 0.0
    assert losses["stroke_valid_count"] == 0
    assert losses["alzheimer_valid_count"] == 0
    assert torch.isfinite(losses["total_loss"]).all()
