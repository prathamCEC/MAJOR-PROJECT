"""
Masked Multi-Task Loss Module for Stroke and Alzheimer's Disease.

Handles missing patient labels via sample-wise masking to prevent data fabrication,
supports class imbalance via positive weighting, and safely balances multi-task objectives.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiTaskConfig, get_default_multitask_config


class MaskedMultiTaskLoss(nn.Module):
    """
    Computes masked multi-task binary cross-entropy loss across Stroke and Alzheimer's prediction.
    """

    def __init__(self, config: Optional[MultiTaskConfig] = None):
        super().__init__()
        self.config = config or get_default_multitask_config()
        self.stroke_weight = self.config.stroke_loss_weight
        self.alzheimer_weight = self.config.alzheimer_loss_weight

        # Optional class imbalance positive weights
        if self.config.stroke_pos_weight is not None:
            self.register_buffer("stroke_pos_weight", torch.tensor([self.config.stroke_pos_weight], dtype=torch.float32))
        else:
            self.stroke_pos_weight = None

        if self.config.alzheimer_pos_weight is not None:
            self.register_buffer("alzheimer_pos_weight", torch.tensor([self.config.alzheimer_pos_weight], dtype=torch.float32))
        else:
            self.alzheimer_pos_weight = None

    def compute_single_task_loss(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute BCE with logits over valid (masked) target samples.

        Args:
            logits: Predicted logits [B, 1] or [B]
            targets: Target labels [B, 1] or [B] (values in {0, 1}; -1 indicates missing)
            mask: Optional boolean or float mask [B, 1] or [B] (1.0 = valid, 0.0 = missing)
            pos_weight: Optional positive class weight

        Returns:
            Tuple of:
            - task_loss: Scalar loss tensor (0.0 if no valid samples)
            - valid_count: Integer number of valid samples
        """
        if targets is None:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), 0

        # Reshape to [B, 1]
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1).float()
        else:
            targets = targets.float()

        # Build validity mask
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1)
            valid_mask = (mask > 0) & (targets >= 0) & torch.isfinite(targets)
        else:
            valid_mask = (targets >= 0) & torch.isfinite(targets)

        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            # Safe zero loss with gradient attachment
            return torch.sum(logits * 0.0), 0

        # Compute per-sample unreduced BCEWithLogits
        sample_losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )

        # Average strictly over valid samples
        masked_loss = torch.sum(sample_losses * valid_mask.float()) / valid_count
        return masked_loss, valid_count

    def forward(
        self,
        stroke_logits: torch.Tensor,
        alzheimer_logits: torch.Tensor,
        stroke_targets: Optional[torch.Tensor] = None,
        alzheimer_targets: Optional[torch.Tensor] = None,
        stroke_mask: Optional[torch.Tensor] = None,
        alzheimer_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            stroke_logits: Predicted stroke logits [B, 1]
            alzheimer_logits: Predicted Alzheimer's logits [B, 1]
            stroke_targets: Ground truth stroke labels [B, 1]
            alzheimer_targets: Ground truth Alzheimer's labels [B, 1]
            stroke_mask: Optional stroke label validity mask [B, 1]
            alzheimer_mask: Optional Alzheimer's label validity mask [B, 1]

        Returns:
            Dict containing:
            - 'total_loss': Combined weighted scalar loss
            - 'stroke_loss': Stroke task scalar loss
            - 'alzheimer_loss': Alzheimer's task scalar loss
            - 'stroke_valid_count': Number of valid stroke samples in batch
            - 'alzheimer_valid_count': Number of valid Alzheimer's samples in batch
        """
        l_stroke, count_stroke = self.compute_single_task_loss(
            logits=stroke_logits,
            targets=stroke_targets,
            mask=stroke_mask,
            pos_weight=self.stroke_pos_weight,
        )

        l_alzheimer, count_alz = self.compute_single_task_loss(
            logits=alzheimer_logits,
            targets=alzheimer_targets,
            mask=alzheimer_mask,
            pos_weight=self.alzheimer_pos_weight,
        )

        total_loss = self.stroke_weight * l_stroke + self.alzheimer_weight * l_alzheimer

        return {
            "total_loss": total_loss,
            "stroke_loss": l_stroke,
            "alzheimer_loss": l_alzheimer,
            "stroke_valid_count": count_stroke,
            "alzheimer_valid_count": count_alz,
        }
