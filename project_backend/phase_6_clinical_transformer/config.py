"""
Configuration Module for Phase 6 FT-Transformer for Clinical Data.

Defines model hyperparameters, transformer layer depths, attention heads,
representation dimensions, and training/inference settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch

from .schema import ClinicalSchema, get_default_retinal_clinical_schema


@dataclass
class ClinicalTransformerConfig:
    """
    Centralized configuration for FT-Transformer and Clinical Representation.
    """
    # Schema specification
    schema: ClinicalSchema = field(default_factory=get_default_retinal_clinical_schema)

    # FT-Transformer architecture hyperparameters
    embed_dim: int = 256          # Embedding dimension per feature token
    num_heads: int = 8            # Multi-head self-attention heads
    num_layers: int = 3           # Transformer blocks stack depth
    ffn_dim: int = 512            # Feed-forward network hidden dimension
    dropout: float = 0.1          # Attention & FFN dropout probability
    ffn_dropout: float = 0.1      # FFN specific dropout
    attention_dropout: float = 0.1

    # Output Clinical Representation (CR) dimension
    # Configured to 512 to seamlessly align with Phase 5 URR (512-dim) for Phase 7
    clinical_representation_dim: int = 512
    pooling_strategy: str = "cls"  # 'cls', 'attention', 'mean'

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30

    # Device & reproducibility
    device: str = "auto"
    random_seed: int = 42

    def get_device(self) -> torch.device:
        """Resolve compute device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def get_default_clinical_config() -> ClinicalTransformerConfig:
    """Return default production ClinicalTransformerConfig."""
    return ClinicalTransformerConfig()


def get_project_root() -> Path:
    """Return project_backend root directory."""
    return Path(__file__).resolve().parent.parent


def get_clinical_outputs_dir() -> Path:
    """Return directory for saving Phase 6 clinical representations and checkpoints."""
    out_dir = get_project_root() / "phase_6_clinical_transformer" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
