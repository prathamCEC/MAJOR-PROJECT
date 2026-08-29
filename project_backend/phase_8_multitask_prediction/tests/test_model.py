"""
Tests for MultiTaskDiseasePredictionNetwork architecture and backward gradients.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from phase_8_multitask_prediction.config import MultiTaskConfig
from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork


def test_model_forward_train_and_eval():
    cfg = MultiTaskConfig(upr_dim=512, shared_hidden_dim=256, task_hidden_dim=128, device="cpu")
    model = MultiTaskDiseasePredictionNetwork(config=cfg)

    B = 4
    upr = torch.randn(B, 512)

    # 1. Training mode forward
    model.train()
    out_train = model(upr, return_probabilities=False)
    assert "stroke_logits" in out_train
    assert "alzheimer_logits" in out_train
    assert out_train["stroke_logits"].shape == (B, 1)
    assert out_train["alzheimer_logits"].shape == (B, 1)
    assert torch.isfinite(out_train["stroke_logits"]).all()
    assert torch.isfinite(out_train["alzheimer_logits"]).all()

    # 2. Evaluation mode forward with probabilities and predictions
    model.eval()
    with torch.no_grad():
        out_eval = model(upr, return_probabilities=True, threshold=0.5)

    assert "stroke_probabilities" in out_eval
    assert "alzheimer_probabilities" in out_eval
    assert "stroke_predictions" in out_eval
    assert "alzheimer_predictions" in out_eval
    assert (out_eval["stroke_probabilities"] >= 0.0).all() and (out_eval["stroke_probabilities"] <= 1.0).all()
    assert (out_eval["alzheimer_probabilities"] >= 0.0).all() and (out_eval["alzheimer_probabilities"] <= 1.0).all()
    assert set(out_eval["stroke_predictions"].unique().tolist()).issubset({0, 1})
    assert set(out_eval["alzheimer_predictions"].unique().tolist()).issubset({0, 1})


def test_model_backward_gradients():
    cfg = MultiTaskConfig(upr_dim=512, shared_hidden_dim=128, task_hidden_dim=64, device="cpu")
    model = MultiTaskDiseasePredictionNetwork(config=cfg)

    B = 2
    upr = torch.randn(B, 512)
    out = model(upr)

    loss = out["stroke_logits"].sum() + out["alzheimer_logits"].sum()
    loss.backward()

    # Verify gradients reach shared trunk and both heads
    has_trunk_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.shared_trunk.parameters())
    has_stroke_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.stroke_head.parameters())
    has_alz_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.alzheimer_head.parameters())

    assert has_trunk_grad, "No gradients in shared trunk"
    assert has_stroke_grad, "No gradients in Stroke head"
    assert has_alz_grad, "No gradients in Alzheimer's head"


def test_model_checkpoint(tmp_path: Path):
    cfg = MultiTaskConfig(upr_dim=256, shared_hidden_dim=64, task_hidden_dim=32, device="cpu")
    model = MultiTaskDiseasePredictionNetwork(config=cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "multitask_ckpt.pth"
    model.save_checkpoint(ckpt_path, optimizer=optimizer, epoch=4)
    assert ckpt_path.exists()

    loaded_model, ckpt_meta = MultiTaskDiseasePredictionNetwork.load_checkpoint(ckpt_path, device="cpu")
    assert ckpt_meta["epoch"] == 4
    assert loaded_model.config.upr_dim == 256
    assert loaded_model.config.shared_hidden_dim == 64
