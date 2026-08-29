"""
Task-Specific Prediction Heads for Stroke and Alzheimer's Disease.

Implements decoupled, independent classification heads producing raw logits during training
and calibrated probabilities & binary predictions during inference.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class DiseasePredictionHead(nn.Module):
    """
    Dedicated Binary Classification Head for a single disease target.
    """

    def __init__(
        self,
        task_name: str,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task_name = task_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        shared_features: torch.Tensor,
        return_probabilities: bool = False,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            shared_features: Tensor [B, input_dim]
            return_probabilities: If True, computes sigmoid probabilities and binary predictions
            threshold: Binary classification threshold (default: 0.5)

        Returns:
            Tuple of:
            - logits: Raw prediction logits [B, 1]
            - probabilities: Sigmoid probabilities [B, 1] (or None if not requested)
            - predictions: Binary 0/1 predictions [B, 1] (or None if not requested)
        """
        logits = self.head(shared_features)  # [B, 1]

        if return_probabilities or not self.training:
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            return logits, probs, preds

        return logits, None, None


class StrokePredictionHead(DiseasePredictionHead):
    """Dedicated Stroke Prediction Head."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__(
            task_name="stroke",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


class AlzheimerPredictionHead(DiseasePredictionHead):
    """Dedicated Alzheimer's Disease Prediction Head."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__(
            task_name="alzheimer",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
