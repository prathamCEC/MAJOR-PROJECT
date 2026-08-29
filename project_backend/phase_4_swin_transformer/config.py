"""
Centralized Configuration for Phase 4 Swin Transformer.

Defines training hyperparameters, model variants, modality configs, and path resolution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .enums import DiseaseTask, Modality

# Model Architecture Defaults
DEFAULT_MODEL_NAME: str = "swin_tiny_patch4_window7_224"
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_NUM_CLASSES: int = 2
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_EPOCHS: int = 20
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-2


@dataclass
class ModalityTrainingConfig:
    """
    Configuration parameters specific to a single modality training pipeline.
    """
    modality: Modality
    is_color: bool
    model_name: str = DEFAULT_MODEL_NAME
    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = DEFAULT_NUM_CLASSES
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    num_workers: int = 0  # Safe Windows multiprocessing default
    random_seed: int = 42
    pretrained: bool = True
    early_stopping_patience: int = 5
    freeze_backbone: bool = False
    unfreeze_at_epoch: int = 0
    mixed_precision: bool = True
    device: str = "auto"
    task: DiseaseTask = DiseaseTask.ALZHEIMERS
    
    # Normalization parameters (ImageNet default for pretrained Swin)
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# OCT-A Configuration: Grayscale capillary vascular features
OCTA_CONFIG = ModalityTrainingConfig(
    modality=Modality.OCTA,
    is_color=False,
    model_name=DEFAULT_MODEL_NAME,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=8,
    epochs=20,
    learning_rate=1e-4,
    weight_decay=1e-2,
)

# OCT-B Configuration: Grayscale cross-sectional retinal layers
OCTB_CONFIG = ModalityTrainingConfig(
    modality=Modality.OCTB,
    is_color=False,
    model_name=DEFAULT_MODEL_NAME,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=8,
    epochs=20,
    learning_rate=1e-4,
    weight_decay=1e-2,
)

# Fundus Configuration: 3-channel RGB retinal photography
FUNDUS_CONFIG = ModalityTrainingConfig(
    modality=Modality.FUNDUS,
    is_color=True,
    model_name=DEFAULT_MODEL_NAME,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=8,
    epochs=20,
    learning_rate=1e-4,
    weight_decay=1e-2,
)

MODALITY_CONFIG_MAP: Dict[Modality, ModalityTrainingConfig] = {
    Modality.OCTA: OCTA_CONFIG,
    Modality.OCTB: OCTB_CONFIG,
    Modality.FUNDUS: FUNDUS_CONFIG,
}


def get_modality_config(modality: Union[str, Modality]) -> ModalityTrainingConfig:
    """Get training config for a given modality."""
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    return MODALITY_CONFIG_MAP[mod_enum]


def get_project_root() -> Path:
    """Get absolute path to project_backend."""
    return Path(__file__).resolve().parent.parent


def get_approved_dataset_dir(modality: Optional[Union[str, Modality]] = None) -> Path:
    """Path to Phase 3 approved datasets."""
    base = get_project_root() / "datasets" / "approved"
    if modality:
        mod_name = modality.value if isinstance(modality, Modality) else str(modality).lower()
        return base / mod_name
    return base


def get_splits_dir() -> Path:
    """Path to dataset split CSV files."""
    splits_dir = get_project_root() / "datasets" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    return splits_dir


def get_outputs_dir(modality: Optional[Union[str, Modality]] = None) -> Path:
    """Path to training checkpoints and experiment outputs."""
    base = get_project_root() / "phase_4_swin_transformer" / "outputs"
    if modality:
        mod_name = modality.value if isinstance(modality, Modality) else str(modality).lower()
        return base / mod_name
    return base
