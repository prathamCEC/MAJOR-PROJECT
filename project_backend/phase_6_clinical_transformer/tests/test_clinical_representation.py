"""
Tests for Clinical Representation Head and full ClinicalFTTransformerModel.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from phase_6_clinical_transformer.config import ClinicalTransformerConfig
from phase_6_clinical_transformer.clinical_representation import ClinicalRepresentationHead
from phase_6_clinical_transformer.clinical_model import ClinicalFTTransformerModel


def test_clinical_representation_head():
    head = ClinicalRepresentationHead(
        embed_dim=256,
        clinical_representation_dim=512,
        pooling_strategy="cls",
    )
    tokens = torch.randn(4, 10, 256)
    cr, cls_token = head(tokens)

    assert cr.shape == (4, 512)
    assert cls_token.shape == (4, 256)
    assert torch.isfinite(cr).all()


def test_clinical_model_forward_and_gradients():
    cfg = ClinicalTransformerConfig(
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        clinical_representation_dim=512,
        device="cpu",
    )
    model = ClinicalFTTransformerModel(config=cfg, categorical_cardinalities=[3, 4, 2])

    B = 4
    x_num = torch.randn(B, cfg.schema.num_numerical)
    x_cat = torch.randint(0, 2, (B, cfg.schema.num_categorical))

    # Forward pass
    out = model(x_num, x_cat)
    assert "clinical_representation" in out
    assert out["clinical_representation"].shape == (B, 512)

    # Backward gradient flow
    target = torch.randn(B, 512)
    loss = nn.MSELoss()(out["clinical_representation"], target)
    loss.backward()

    # Check gradients exist
    has_tok_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.tokenizer.parameters())
    has_trans_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.transformer.parameters())
    has_head_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.representation_head.parameters())

    assert has_tok_grad, "No gradients in Tokenizer"
    assert has_trans_grad, "No gradients in FT-Transformer backbone"
    assert has_head_grad, "No gradients in Representation Head"


def test_clinical_model_checkpoint(tmp_path: Path):
    from phase_6_clinical_transformer.schema import ClinicalSchema
    schema = ClinicalSchema(
        numerical_features=["feat_num"],
        categorical_features=["cat_1", "cat_2", "cat_3"],
        binary_features=[],
    )
    cfg = ClinicalTransformerConfig(
        schema=schema,
        embed_dim=64,
        clinical_representation_dim=256,
        device="cpu",
    )
    model = ClinicalFTTransformerModel(config=cfg, categorical_cardinalities=[3, 3, 3])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "clinical_ckpt.pth"
    model.save_checkpoint(ckpt_path, optimizer=optimizer, epoch=3)
    assert ckpt_path.exists()

    loaded_model, ckpt_meta = ClinicalFTTransformerModel.load_checkpoint(ckpt_path, device="cpu")
    assert ckpt_meta["epoch"] == 3
    assert loaded_model.config.embed_dim == 64
    assert loaded_model.config.clinical_representation_dim == 256
