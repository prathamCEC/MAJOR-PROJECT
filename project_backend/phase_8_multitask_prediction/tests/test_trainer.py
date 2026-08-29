"""
Tests for MultiTaskTrainer training epoch and evaluation routines.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from phase_8_multitask_prediction.config import MultiTaskConfig
from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from phase_8_multitask_prediction.trainer import MultiTaskTrainer


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, upr, stroke_label, alzheimer_label):
        self.upr = upr
        self.stroke_label = stroke_label
        self.alzheimer_label = alzheimer_label

    def __len__(self):
        return len(self.upr)

    def __getitem__(self, idx):
        return {
            "upr": self.upr[idx],
            "stroke_label": self.stroke_label[idx],
            "alzheimer_label": self.alzheimer_label[idx],
        }


def test_trainer_epoch_and_eval():
    cfg = MultiTaskConfig(upr_dim=512, shared_hidden_dim=128, task_hidden_dim=64, device="cpu")
    model = MultiTaskDiseasePredictionNetwork(config=cfg)
    trainer = MultiTaskTrainer(model=model, config=cfg)

    # 16 samples synthetic dataset
    upr = torch.randn(16, 512)
    st_labels = torch.randint(0, 2, (16, 1)).float()
    al_labels = torch.randint(0, 2, (16, 1)).float()

    dataset = DictDataset(upr, st_labels, al_labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train 1 epoch
    train_res = trainer.train_epoch(loader)
    assert "train_total_loss" in train_res
    assert train_res["train_total_loss"] > 0.0

    # Evaluate
    eval_res = trainer.evaluate(loader)
    assert "val_total_loss" in eval_res
    assert "stroke_accuracy" in eval_res
    assert "alzheimer_accuracy" in eval_res
