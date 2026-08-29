"""
Configuration Module for Phase 9 Monte Carlo Dropout & Uncertainty Estimation.

Defines stochastic sample counts, decision thresholds, entropy numerical epsilons,
uncertainty scaling factors, and compute configurations.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch


@dataclass
class UncertaintyConfig:
    """
    Centralized configuration for Phase 9 Monte Carlo Dropout & Model Confidence.
    """
    # Monte Carlo sampling parameters
    mc_samples: int = 30                    # Number of stochastic forward passes (default: 30)
    classification_threshold: float = 0.5  # Decision threshold for binary classification
    epsilon: float = 1e-7                   # Numerical safety epsilon for log entropy
    uncertainty_scale: float = 0.25         # Max Bernoulli variance (0.5 * (1-0.5) = 0.25)
    store_mc_predictions: bool = False     # If True, stores individual [B, T] passes

    # Device & compute
    device: str = "auto"
    random_seed: Optional[int] = None       # Optional global seed

    def get_device(self) -> torch.device:
        """Resolve compute device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.mc_samples < 2:
            raise ValueError(
                f"Invalid mc_samples={self.mc_samples}. Monte Carlo Dropout uncertainty estimation "
                f"requires at least 2 stochastic passes."
            )
        if not (0.0 < self.classification_threshold < 1.0):
            raise ValueError(
                f"Invalid classification_threshold={self.classification_threshold}. Must be in (0, 1)."
            )
        if self.uncertainty_scale <= 0:
            raise ValueError(f"uncertainty_scale must be positive, got {self.uncertainty_scale}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        return cls(**data)

    def save_json(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "UncertaintyConfig":
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_uncertainty_config() -> UncertaintyConfig:
    """Return default production UncertaintyConfig."""
    return UncertaintyConfig()


def get_project_root() -> Path:
    """Return project_backend root directory."""
    return Path(__file__).resolve().parent.parent


def get_phase9_outputs_dir() -> Path:
    """Return directory for saving Phase 9 uncertainty estimates."""
    out_dir = get_project_root() / "phase_9_uncertainty" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
