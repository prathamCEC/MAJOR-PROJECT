"""
Tests for full RetinaClinicalFusionModel forward pass, gradients, and checkpointing.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from phase_7_retina_clinical_fusion.config import RetinaClinicalConfig
from phase_7_retina_clinical_fusion.fusion_model import RetinaClinicalFusionModel


def test_fusion_model_forward_and_gradients():
    cfg = RetinaClinicalConfig(
        retinal_input_dim=512,
        clinical_input_dim=512,
        common_embed_dim=256,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
        upr_dim=512,
        device="cpu",
    )
    model = RetinaClinicalFusionModel(config=cfg)

    B = 4
    retinal = torch.randn(B, 512)
    clinical = torch.randn(B, 512)

    # 1. Forward Pass
    out = model(retinal_representation=retinal, clinical_representation=clinical)
    assert "upr" in out
    assert out["upr"].shape == (B, 512)
    assert torch.isfinite(out["upr"]).all()

    # 2. Backward Gradient Flow
    target = torch.randn(B, 512)
    loss = nn.MSELoss()(out["upr"], target)
    loss.backward()

    # Check gradients exist
    has_proj_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.projection.parameters())
    has_attn_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.cross_attention.parameters())
    has_fuse_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.fusion.parameters())

    assert has_proj_grad, "No gradients in Projection layer"
    assert has_attn_grad, "No gradients in Cross-Attention Transformer"
    assert has_fuse_grad, "No gradients in Fusion / UPR Head"


def test_fusion_model_checkpoint(tmp_path: Path):
    cfg = RetinaClinicalConfig(common_embed_dim=128, upr_dim=256, device="cpu")
    model = RetinaClinicalFusionModel(config=cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "phase7_ckpt.pth"
    model.save_checkpoint(ckpt_path, optimizer=optimizer, epoch=5)
    assert ckpt_path.exists()

    loaded_model, ckpt_meta = RetinaClinicalFusionModel.load_checkpoint(ckpt_path, device="cpu")
    assert ckpt_meta["epoch"] == 5
    assert loaded_model.config.common_embed_dim == 128
    assert loaded_model.config.upr_dim == 256
