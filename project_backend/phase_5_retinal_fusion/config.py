"""
Configuration Module for Phase 5 Retinal Multimodal Fusion.

Defines hyperparameters, dimensions, attention heads, dropout rates, and paths
for Dynamic Modality Reliability Attention (DMRA) and Unified Retinal Representation (URR).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch


@dataclass
class FusionConfig:
    """
    Centralized configuration for Phase 5 Retinal Multimodal Fusion.
    """
    # Supported modalities
    modalities: List[str] = field(default_factory=lambda: ["octa", "octb", "fundus"])

    # Modality input feature dimensions (from Phase 4 Swin backbones)
    # Default Swin-Tiny feature dimension is 768.
    input_dims: Dict[str, int] = field(default_factory=lambda: {
        "octa": 768,
        "octb": 768,
        "fundus": 768,
    })

    # Common projection embedding dimension for multimodal fusion
    embed_dim: int = 512

    # Transformer Cross-Attention parameters
    num_heads: int = 8
    num_fusion_layers: int = 2
    ffn_dim: int = 1024
    dropout: float = 0.1

    # Dynamic Modality Reliability Attention (DMRA) parameters
    reliability_hidden_dim: int = 256
    reliability_temperature: float = 1.0

    # Unified Retinal Representation (URR) output dimension
    urr_dim: int = 512
    urr_pooling: str = "attention"  # Options: 'attention', 'mean', 'cls'

    # Compute device ('auto', 'cuda', 'cpu')
    device: str = "auto"

    # Reproducibility
    random_seed: int = 42

    def get_device(self) -> torch.device:
        """Resolve torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def get_default_fusion_config() -> FusionConfig:
    """Return default production FusionConfig instance."""
    return FusionConfig()


def get_project_root() -> Path:
    """Return project_backend root directory."""
    return Path(__file__).resolve().parent.parent


def get_fusion_outputs_dir() -> Path:
    """Return directory for saving Phase 5 fusion checkpoints and exports."""
    out_dir = get_project_root() / "phase_5_retinal_fusion" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
