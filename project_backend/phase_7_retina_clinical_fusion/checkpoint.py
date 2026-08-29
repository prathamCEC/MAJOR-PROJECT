"""
Checkpoint Management Module for Phase 7 Retina-Clinical Fusion.

Handles atomic saving and restoration of model weights, configurations, and metadata.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from .config import RetinaClinicalConfig
from .fusion_model import RetinaClinicalFusionModel


class Phase7CheckpointManager:
    """
    Manages checkpoints for Phase 7 Retina-Clinical Fusion models.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_phase7_model.pth"
        self.last_model_path = self.checkpoint_dir / "last_phase7_model.pth"

    def save_checkpoint(
        self,
        model: RetinaClinicalFusionModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model weights and metadata.
        """
        path = self.best_model_path if is_best else self.last_model_path
        saved_path = model.save_checkpoint(
            output_path=path,
            optimizer=optimizer,
            epoch=epoch,
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
    ) -> Tuple[RetinaClinicalFusionModel, Dict[str, Any]]:
        """
        Load model and metadata from checkpoint.
        """
        return RetinaClinicalFusionModel.load_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            device=device,
        )
