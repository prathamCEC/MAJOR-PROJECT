"""
Tests for RetinalMultimodalFusionModel: forward pass, backward gradients, and checkpoints.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from phase_5_retinal_fusion.config import FusionConfig
from phase_5_retinal_fusion.fusion_model import RetinalMultimodalFusionModel


def test_fusion_model_forward_and_gradients():
    cfg = FusionConfig(
        embed_dim=256,
        num_heads=4,
        num_fusion_layers=2,
        ffn_dim=512,
        urr_dim=256,
        device="cpu",
    )
    model = RetinalMultimodalFusionModel(config=cfg)

    batch_size = 2
    feats = {
        "octa": torch.randn(batch_size, 49, 768),
        "octb": torch.randn(batch_size, 49, 768),
        "fundus": torch.randn(batch_size, 768),
    }

    # Forward pass
    out = model(feats)

    assert "urr" in out
    assert out["urr"].shape == (batch_size, 256)
    assert "modality_weights" in out
    assert len(out["modality_weights"]) == 3

    # Check backward gradient flow through entire model
    target = torch.randn(batch_size, 256)
    loss = nn.MSELoss()(out["urr"], target)
    loss.backward()

    # Verify gradients in DMRA, Cross-Attention, and Projections
    has_proj_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.projections.parameters())
    has_dmra_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.dmra.parameters())
    has_cross_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.cross_attention.parameters())
    has_urr_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.urr_head.parameters())

    assert has_proj_grad, "No gradients in Projection layers"
    assert has_dmra_grad, "No gradients in DMRA reliability module"
    assert has_cross_grad, "No gradients in Cross-Attention module"
    assert has_urr_grad, "No gradients in URR Head module"


def test_fusion_model_checkpoint(tmp_path: Path):
    cfg = FusionConfig(embed_dim=128, urr_dim=128, device="cpu")
    model = RetinalMultimodalFusionModel(config=cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "fusion_checkpoint.pt"
    saved = model.save_checkpoint(ckpt_path, optimizer=optimizer, epoch=5)
    assert saved.exists()

    # Load into new model instance
    new_opt = optim.AdamW(model.parameters(), lr=1e-3)
    loaded_model, ckpt_meta = RetinalMultimodalFusionModel.load_checkpoint(
        ckpt_path, optimizer=new_opt, device="cpu"
    )

    assert ckpt_meta["epoch"] == 5
    assert loaded_model.config.embed_dim == 128
    assert loaded_model.config.urr_dim == 128
