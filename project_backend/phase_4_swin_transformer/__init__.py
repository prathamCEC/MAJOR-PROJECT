"""
Phase 4: Swin Transformer for Retinal Disease Analysis (OCT-A, OCT-B, Fundus).

Provides modality-aware deep learning architectures, dataset pipelines, training,
evaluation, inference, and explainability for Stroke and Alzheimer's disease retinal patterns.
"""

from .enums import DiseaseTask, Modality, PredictionOutput, SplitType
from .config import (
    ModalityTrainingConfig,
    OCTA_CONFIG,
    OCTB_CONFIG,
    FUNDUS_CONFIG,
    get_modality_config,
    get_approved_dataset_dir,
    get_splits_dir,
    get_outputs_dir,
)
from .dataset import RetinalDataset, RetinalItem, create_dataloader
from .transforms import get_transforms
from .models.swin_factory import SwinRetinalClassifier, create_swin_model
from .models.octa_model import OCTASwinModel
from .models.octb_model import OCTBSwinModel
from .models.fundus_model import FundusSwinModel
from .metrics import (
    EvaluationMetrics,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from .checkpoint import CheckpointManager
from .validation import DatasetReport, DatasetValidator, validate_modality_dataset
from .leakage_check import LeakageCheckResult, check_splits_leakage
from .split_dataset import create_dataset_splits, load_dataset_splits
from .train import train_swin
from .evaluate import evaluate_checkpoint
from .inference import SwinInferenceEngine
from .explainability import SwinExplainabilityEngine
from .utils import set_seed, get_device, create_experiment_dir, compute_class_weights

__version__ = "1.0.0"

__all__ = [
    "Modality",
    "DiseaseTask",
    "SplitType",
    "PredictionOutput",
    "ModalityTrainingConfig",
    "OCTA_CONFIG",
    "OCTB_CONFIG",
    "FUNDUS_CONFIG",
    "get_modality_config",
    "get_approved_dataset_dir",
    "get_splits_dir",
    "get_outputs_dir",
    "RetinalDataset",
    "RetinalItem",
    "create_dataloader",
    "get_transforms",
    "SwinRetinalClassifier",
    "create_swin_model",
    "OCTASwinModel",
    "OCTBSwinModel",
    "FundusSwinModel",
    "EvaluationMetrics",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "CheckpointManager",
    "DatasetReport",
    "DatasetValidator",
    "validate_modality_dataset",
    "LeakageCheckResult",
    "check_splits_leakage",
    "create_dataset_splits",
    "load_dataset_splits",
    "train_swin",
    "evaluate_checkpoint",
    "SwinInferenceEngine",
    "SwinExplainabilityEngine",
    "set_seed",
    "get_device",
    "create_experiment_dir",
    "compute_class_weights",
]
