"""
Configuration Module for Phase 8 Multi-Task Disease Prediction Network.

Defines model hyperparameters, shared representation dimensions, task loss weights,
classification thresholds, optimizer settings, and compute configurations.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch


@dataclass
class MultiTaskConfig:
    """
    Centralized configuration for Phase 8 Multi-Task Disease Prediction Network.
    """
    # Input representation dimension (from Phase 7 UPR)
    upr_dim: int = 512

    # Shared prediction trunk architecture
    shared_hidden_dim: int = 256
    task_hidden_dim: int = 128
    dropout: float = 0.2

    # Multi-task loss balancing
    stroke_loss_weight: float = 1.0        # lambda_stroke
    alzheimer_loss_weight: float = 1.0     # lambda_alzheimer
    stroke_pos_weight: Optional[float] = None
    alzheimer_pos_weight: Optional[float] = None

    # Decision threshold (configurable, default 0.5)
    # Research prototype threshold (not clinically validated)
    classification_threshold: float = 0.5

    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    batch_size: int = 16

    # Device & reproducibility
    device: str = "auto"
    random_seed: int = 42

    def get_device(self) -> torch.device:
        """Resolve compute device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiTaskConfig":
        return cls(**data)

    def save_json(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "MultiTaskConfig":
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_multitask_config() -> MultiTaskConfig:
    """Return default production MultiTaskConfig."""
    return MultiTaskConfig()


def get_project_root() -> Path:
    """Return project_backend root directory."""
    return Path(__file__).resolve().parent.parent


def get_phase8_outputs_dir() -> Path:
    """Return directory for saving Phase 8 model outputs and checkpoints."""
    out_dir = get_project_root() / "phase_8_multitask_prediction" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
