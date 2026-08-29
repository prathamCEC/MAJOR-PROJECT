"""
Tests for CheckpointManager.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from phase_4_swin_transformer.enums import DiseaseTask, Modality
from phase_4_swin_transformer.checkpoint import CheckpointManager


def test_checkpoint_save_and_load(tmp_path: Path):
    mgr = CheckpointManager(tmp_path)
    model = nn.Linear(10, 2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    class_mapping = {"class_0": 0, "class_1": 1}

    # Save checkpoint
    saved_p = mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        best_metric=0.85,
        class_mapping=class_mapping,
        modality=Modality.OCTA,
        task=DiseaseTask.ALZHEIMERS,
        is_best=True,
    )

    assert saved_p.exists()
    assert mgr.best_model_path.exists()
    assert mgr.last_model_path.exists()

    # Load into new model
    new_model = nn.Linear(10, 2)
    new_opt = optim.AdamW(new_model.parameters(), lr=1e-3)
    ckpt = CheckpointManager.load_checkpoint(mgr.best_model_path, new_model, new_opt)

    assert ckpt["epoch"] == 3
    assert ckpt["best_metric"] == 0.85
    assert ckpt["modality"] == "octa"
    assert ckpt["task"] == "alzheimers"
    assert ckpt["class_mapping"] == class_mapping

    # Verify weights match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
