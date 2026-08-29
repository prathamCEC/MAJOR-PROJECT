"""
Trainer Module for Phase 8 Multi-Task Disease Prediction Network.

Implements supervised training and validation routines with masked multi-task loss,
gradient clipping, and metrics logging.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import MultiTaskConfig, get_default_multitask_config
from .model import MultiTaskDiseasePredictionNetwork
from .loss import MaskedMultiTaskLoss
from .metrics import MultiTaskMetricsCalculator


class MultiTaskTrainer:
    """
    Supervised Training and Evaluation Manager for Multi-Task Disease Prediction.
    """

    def __init__(
        self,
        model: MultiTaskDiseasePredictionNetwork,
        config: Optional[MultiTaskConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.config = config or get_default_multitask_config()
        self.device = self.config.get_device()
        self.model = model.to(self.device)
        self.loss_fn = MaskedMultiTaskLoss(config=self.config).to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Execute one training epoch.
        """
        self.model.train()
        total_loss_accum = 0.0
        stroke_loss_accum = 0.0
        alz_loss_accum = 0.0
        n_batches = 0

        for batch in dataloader:
            upr = batch["upr"].to(self.device)
            stroke_target = batch.get("stroke_label")
            alz_target = batch.get("alzheimer_label")
            stroke_mask = batch.get("stroke_mask")
            alz_mask = batch.get("alzheimer_mask")

            if stroke_target is not None:
                stroke_target = stroke_target.to(self.device)
            if alz_target is not None:
                alz_target = alz_target.to(self.device)
            if stroke_mask is not None:
                stroke_mask = stroke_mask.to(self.device)
            if alz_mask is not None:
                alz_mask = alz_mask.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(upr, return_probabilities=False)

            losses = self.loss_fn(
                stroke_logits=out["stroke_logits"],
                alzheimer_logits=out["alzheimer_logits"],
                stroke_targets=stroke_target,
                alzheimer_targets=alz_target,
                stroke_mask=stroke_mask,
                alzheimer_mask=alz_mask,
            )

            total_loss = losses["total_loss"]
            if total_loss.requires_grad:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss_accum += total_loss.item()
            stroke_loss_accum += losses["stroke_loss"].item()
            alz_loss_accum += losses["alzheimer_loss"].item()
            n_batches += 1

        return {
            "train_total_loss": total_loss_accum / max(1, n_batches),
            "train_stroke_loss": stroke_loss_accum / max(1, n_batches),
            "train_alzheimer_loss": alz_loss_accum / max(1, n_batches),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on validation dataloader.
        """
        self.model.eval()
        all_stroke_logits, all_stroke_probs, all_stroke_preds, all_stroke_targets = [], [], [], []
        all_alz_logits, all_alz_probs, all_alz_preds, all_alz_targets = [], [], [], []
        total_loss_accum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                upr = batch["upr"].to(self.device)
                stroke_target = batch.get("stroke_label")
                alz_target = batch.get("alzheimer_label")

                out = self.model(upr, return_probabilities=True)

                if stroke_target is not None:
                    stroke_target = stroke_target.to(self.device)
                    all_stroke_targets.append(stroke_target.cpu())
                    all_stroke_logits.append(out["stroke_logits"].cpu())
                    all_stroke_probs.append(out["stroke_probabilities"].cpu())
                    all_stroke_preds.append(out["stroke_predictions"].cpu())

                if alz_target is not None:
                    alz_target = alz_target.to(self.device)
                    all_alz_targets.append(alz_target.cpu())
                    all_alz_logits.append(out["alzheimer_logits"].cpu())
                    all_alz_probs.append(out["alzheimer_probabilities"].cpu())
                    all_alz_preds.append(out["alzheimer_predictions"].cpu())

                losses = self.loss_fn(
                    stroke_logits=out["stroke_logits"],
                    alzheimer_logits=out["alzheimer_logits"],
                    stroke_targets=stroke_target,
                    alzheimer_targets=alz_target,
                )
                total_loss_accum += losses["total_loss"].item()
                n_batches += 1

        val_metrics = {
            "val_total_loss": total_loss_accum / max(1, n_batches),
        }

        # Compute full task metrics
        st_true = torch.cat(all_stroke_targets, dim=0) if all_stroke_targets else None
        st_pred = torch.cat(all_stroke_preds, dim=0) if all_stroke_preds else None
        st_prob = torch.cat(all_stroke_probs, dim=0) if all_stroke_probs else None

        al_true = torch.cat(all_alz_targets, dim=0) if all_alz_targets else None
        al_pred = torch.cat(all_alz_preds, dim=0) if all_alz_preds else None
        al_prob = torch.cat(all_alz_probs, dim=0) if all_alz_probs else None

        task_metrics = MultiTaskMetricsCalculator.calculate_multitask_metrics(
            stroke_true=st_true,
            stroke_pred=st_pred,
            stroke_prob=st_prob,
            alzheimer_true=al_true,
            alzheimer_pred=al_pred,
            alzheimer_prob=al_prob,
        )
        val_metrics.update(task_metrics)
        return val_metrics
