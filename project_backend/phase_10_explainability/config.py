"""
Configuration Module for Phase 10 Model Explainability (Grad-CAM + SHAP).

Defines target layers, sampling sizes, colormaps, output paths, and compute configurations.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch


@dataclass
class ExplainabilityConfig:
    """
    Centralized configuration for Phase 10 Multimodal Explainability.
    """
    # Grad-CAM settings
    gradcam_enabled: bool = True
    gradcam_target_layer: Optional[str] = None  # None = auto-detect last Swin stage
    gradcam_colormap: str = "jet"
    gradcam_alpha: float = 0.5                 # Overlay transparency (0.0 = only image, 1.0 = only heatmap)

    # SHAP settings
    shap_enabled: bool = True
    shap_background_samples: int = 20          # Reference samples for background baseline
    shap_max_samples: int = 50                 # Evaluation sample limit
    shap_epsilon: float = 1e-6

    # Monte Carlo Uncertainty Integration (Phase 9)
    include_phase9_uncertainty: bool = True
    mc_samples: int = 15                       # Fast MC sampling for explainability passes

    # Device & output paths
    device: str = "auto"
    save_visualizations: bool = True
    output_dir: Optional[str] = None

    def get_device(self) -> torch.device:
        """Resolve compute device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_output_dir(self) -> Path:
        """Resolve output directory."""
        if self.output_dir:
            path = Path(self.output_dir).resolve()
        else:
            path = get_project_root() / "phase_10_explainability" / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplainabilityConfig":
        return cls(**data)

    def save_json(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "ExplainabilityConfig":
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_explainability_config() -> ExplainabilityConfig:
    """Return default production ExplainabilityConfig."""
    return ExplainabilityConfig()


def get_project_root() -> Path:
    """Return project_backend root directory."""
    return Path(__file__).resolve().parent.parent


def get_phase10_outputs_dir() -> Path:
    """Return directory for saving Phase 10 heatmaps and reports."""
    out_dir = get_project_root() / "phase_10_explainability" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
