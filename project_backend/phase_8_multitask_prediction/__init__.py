"""
Phase 8: Multi-Task Disease Prediction Network (Stroke + Alzheimer's Disease).

Consumes the Unified Patient Representation (Phase 7 UPR) and performs simultaneous,
decoupled binary classification for Stroke and Alzheimer's disease.
"""

from .config import (
    MultiTaskConfig,
    get_default_multitask_config,
    get_phase8_outputs_dir,
)
from .shared_network import SharedPredictionTrunk
from .prediction_heads import (
    DiseasePredictionHead,
    StrokePredictionHead,
    AlzheimerPredictionHead,
)
from .loss import MaskedMultiTaskLoss
from .model import MultiTaskDiseasePredictionNetwork
from .metrics import MultiTaskMetricsCalculator
from .trainer import MultiTaskTrainer
from .validation import (
    validate_prediction_inputs,
    validate_prediction_outputs,
)
from .checkpoint import Phase8CheckpointManager
from .inference import EndToEndDiseasePredictor

__version__ = "1.0.0"

__all__ = [
    "MultiTaskConfig",
    "get_default_multitask_config",
    "get_phase8_outputs_dir",
    "SharedPredictionTrunk",
    "DiseasePredictionHead",
    "StrokePredictionHead",
    "AlzheimerPredictionHead",
    "MaskedMultiTaskLoss",
    "MultiTaskDiseasePredictionNetwork",
    "MultiTaskMetricsCalculator",
    "MultiTaskTrainer",
    "validate_prediction_inputs",
    "validate_prediction_outputs",
    "Phase8CheckpointManager",
    "EndToEndDiseasePredictor",
]
