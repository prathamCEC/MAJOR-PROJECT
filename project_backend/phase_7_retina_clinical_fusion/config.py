"""
Configuration Module for Phase 7 Retina-Clinical Cross-Attention Fusion & Unified Patient Representation (UPR).

Defines architectural hyperparameters, projection dimensions, cross-attention depth,
fusion gating strategies, and compute configurations.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch


@dataclass
class RetinaClinicalConfig:
    """
    Centralized configuration for Phase 7 Retina-Clinical Cross-Attention & UPR.
    """
    # Input representation dimensions
    retinal_input_dim: int = 512    # Phase 5 URR output dimension (or token dim)
    clinical_input_dim: int = 512   # Phase 6 Clinical Representation dimension (or token dim)

    # Common latent cross-attention space
    common_embed_dim: int = 512     # Dimension of shared multimodal interaction space
    num_heads: int = 8              # Number of cross-attention heads
    num_layers: int = 2             # Depth of bidirectional cross-attention transformer blocks
    ffn_dim: int = 1024             # Feed-Forward Network hidden dimension
    dropout: float = 0.1            # Transformer dropout probability
    attention_dropout: float = 0.1  # Cross-attention specific dropout

    # Output Unified Patient Representation (UPR) dimension
    upr_dim: int = 512              # Fixed output dimension consumed by downstream Phase 8
    fusion_strategy: str = "gated"  # 'gated', 'concat_proj', 'cross_attention_cls'
    pooling_strategy: str = "attentive"  # 'attentive', 'mean', 'cls'

    # Compute & device
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
    def from_dict(cls, data: Dict[str, Any]) -> "RetinaClinicalConfig":
        return cls(**data)

    def save_json(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "RetinaClinicalConfig":
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_retina_clinical_config() -> RetinaClinicalConfig:
    """Return default production RetinaClinicalConfig."""
    return RetinaClinicalConfig()


def get_project_root() -> Path:
    """Return project_backend root directory."""
    return Path(__file__).resolve().parent.parent


def get_phase7_outputs_dir() -> Path:
    """Return directory for saving Phase 7 UPR vectors and checkpoints."""
    out_dir = get_project_root() / "phase_7_retina_clinical_fusion" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
