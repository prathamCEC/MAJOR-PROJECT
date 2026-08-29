"""
End-to-End Multi-Task Disease Prediction Network.

Combines Shared Prediction Representation Trunk with independent classification heads for
Stroke and Alzheimer's Disease prediction. Designed for Phase 9 Monte Carlo Dropout compatibility.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .config import MultiTaskConfig, get_default_multitask_config
from .shared_network import SharedPredictionTrunk
from .prediction_heads import StrokePredictionHead, AlzheimerPredictionHead


class MultiTaskDiseasePredictionNetwork(nn.Module):
    """
    Complete Phase 8 Multi-Task Architecture for Multimodal Retinal-Clinical Disease Prediction.
    """

    def __init__(self, config: Optional[MultiTaskConfig] = None):
        super().__init__()
        self.config = config or get_default_multitask_config()

        # 1. Shared Prediction Representation Trunk
        self.shared_trunk = SharedPredictionTrunk(
            upr_dim=self.config.upr_dim,
            shared_hidden_dim=self.config.shared_hidden_dim,
            dropout=self.config.dropout,
        )

        # 2. Independent Disease Prediction Heads
        self.stroke_head = StrokePredictionHead(
            input_dim=self.config.shared_hidden_dim,
            hidden_dim=self.config.task_hidden_dim,
            dropout=self.config.dropout,
        )

        self.alzheimer_head = AlzheimerPredictionHead(
            input_dim=self.config.shared_hidden_dim,
            hidden_dim=self.config.task_hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        upr: torch.Tensor,
        return_probabilities: bool = False,
        threshold: Optional[float] = None,
        enable_mc_dropout: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            upr: Unified Patient Representation tensor [B, upr_dim]
            return_probabilities: Whether to compute sigmoid probabilities & binary predictions
            threshold: Classification threshold (default: config.classification_threshold)
            enable_mc_dropout: If True, keeps dropout active during evaluation for Phase 9 MC-Dropout

        Returns:
            Dict containing:
            - 'stroke_logits': Raw Stroke logits [B, 1]
            - 'alzheimer_logits': Raw Alzheimer's logits [B, 1]
            - 'stroke_probabilities': Stroke probabilities [B, 1] (if requested or in eval mode)
            - 'alzheimer_probabilities': Alzheimer's probabilities [B, 1] (if requested or in eval mode)
            - 'stroke_predictions': Binary Stroke predictions [B, 1] (0 or 1)
            - 'alzheimer_predictions': Binary Alzheimer's predictions [B, 1] (0 or 1)
            - 'shared_features': Shared multimodal latent vector [B, shared_hidden_dim]
        """
        thresh = threshold if threshold is not None else self.config.classification_threshold

        # Phase 9 compatibility: Enable stochastic dropout forward passes during evaluation
        if enable_mc_dropout:
            self._set_dropout_train(True)

        # 1. Shared Representation Processing
        shared_feat = self.shared_trunk(upr)  # [B, shared_hidden_dim]

        # 2. Task-Specific Heads
        stroke_logits, stroke_probs, stroke_preds = self.stroke_head(
            shared_features=shared_feat,
            return_probabilities=return_probabilities,
            threshold=thresh,
        )

        alz_logits, alz_probs, alz_preds = self.alzheimer_head(
            shared_features=shared_feat,
            return_probabilities=return_probabilities,
            threshold=thresh,
        )

        out = {
            "stroke_logits": stroke_logits,
            "alzheimer_logits": alz_logits,
            "shared_features": shared_feat,
        }

        if stroke_probs is not None:
            out["stroke_probabilities"] = stroke_probs
            out["stroke_predictions"] = stroke_preds

        if alz_probs is not None:
            out["alzheimer_probabilities"] = alz_probs
            out["alzheimer_predictions"] = alz_preds

        return out

    def _set_dropout_train(self, mode: bool = True) -> None:
        """Helper to specifically enable/disable dropout layers for Phase 9 Monte Carlo analysis."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(mode)

    def save_checkpoint(
        self,
        output_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model state, config, and training metadata."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics or {},
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(payload, str(path))
        return path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Tuple["MultiTaskDiseasePredictionNetwork", Dict[str, Any]]:
        """Load model from checkpoint."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        try:
            ckpt = torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location=device)

        cfg_dict = ckpt.get("config", {})
        config = MultiTaskConfig.from_dict(cfg_dict)

        model = cls(config=config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(torch.device(device))

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return model, ckpt
