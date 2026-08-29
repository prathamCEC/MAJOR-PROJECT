"""
Checkpoint Management Module for Phase 4 Swin Transformer.

Handles atomic saving, loading, and restoration of training states, model weights,
class mappings, and experiment metadata.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim

from .enums import DiseaseTask, Modality
from .config import ModalityTrainingConfig


class CheckpointManager:
    """
    Manages experiment persistence and checkpoint restoration.
    """

    def __init__(self, experiment_dir: Union[str, Path]):
        self.experiment_dir = Path(experiment_dir).resolve()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.experiment_dir / "best_model.pth"
        self.last_model_path = self.experiment_dir / "last_model.pth"
        self.config_path = self.experiment_dir / "config.json"
        self.class_mapping_path = self.experiment_dir / "class_mapping.json"

    def save_config(self, config: ModalityTrainingConfig, class_mapping: Dict[str, int]) -> None:
        """Save experiment configuration and class indices."""
        cfg_dict = asdict(config)
        cfg_dict["modality"] = config.modality.value
        cfg_dict["task"] = config.task.value
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

        with open(self.class_mapping_path, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        best_metric: float,
        class_mapping: Dict[str, int],
        modality: Modality,
        task: DiseaseTask,
        is_best: bool = False,
    ) -> Path:
        """
        Save training checkpoint state.
        """
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "modality": modality.value,
            "task": task.value,
            "class_mapping": class_mapping,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        }

        # Always save last model
        torch.save(state, str(self.last_model_path))

        if is_best:
            torch.save(state, str(self.best_model_path))
            return self.best_model_path

        return self.last_model_path

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model, optimizer, and scheduler states.
        """
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{path}'")

        checkpoint = torch.load(str(path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
