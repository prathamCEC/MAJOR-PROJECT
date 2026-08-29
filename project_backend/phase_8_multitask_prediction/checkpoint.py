"""
Checkpoint Management Module for Phase 8 Multi-Task Network.

Handles atomic saving and restoration of model weights, optimizer states,
training configs, and validation metrics.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from .config import MultiTaskConfig
from .model import MultiTaskDiseasePredictionNetwork


class Phase8CheckpointManager:
    """
    Manages checkpoints for Phase 8 Multi-Task Disease Prediction Network.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_multitask_model.pth"
        self.last_model_path = self.checkpoint_dir / "last_multitask_model.pth"

    def save_checkpoint(
        self,
        model: MultiTaskDiseasePredictionNetwork,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model weights, optimizer, and metadata."""
        path = self.best_model_path if is_best else self.last_model_path
        saved_path = model.save_checkpoint(
            output_path=path,
            optimizer=optimizer,
            epoch=epoch,
            val_metrics=val_metrics,
            metadata=extra_metadata,
        )

        if is_best and self.last_model_path != self.best_model_path:
            import shutil
            shutil.copy(str(saved_path), str(self.last_model_path))

        return saved_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Tuple[MultiTaskDiseasePredictionNetwork, Dict[str, Any]]:
        """Load model from checkpoint."""
        return MultiTaskDiseasePredictionNetwork.load_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            device=device,
        )
