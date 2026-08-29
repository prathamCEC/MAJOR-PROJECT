"""
Phase 6: FT-Transformer for Structured Clinical Data & Clinical Representation (CR).

Provides schema configuration, tabular preprocessing, feature tokenization,
and multi-layer Transformer modeling for patient clinical attributes.
"""

from .schema import ClinicalSchema, get_default_retinal_clinical_schema
from .config import (
    ClinicalTransformerConfig,
    get_default_clinical_config,
    get_clinical_outputs_dir,
)
from .preprocessing import ClinicalPreprocessor, PreprocessorState
from .feature_tokenizer import (
    NumericalFeatureTokenizer,
    CategoricalFeatureTokenizer,
    ClinicalFeatureTokenizer,
)
from .ft_transformer import FTTransformerBlock, FTTransformerBackbone
from .clinical_representation import AttentiveTokenPooler, ClinicalRepresentationHead
from .clinical_model import ClinicalFTTransformerModel
from .dataset import (
    ClinicalTabularDataset,
    create_clinical_dataloader,
    patient_level_split,
)
from .feature_loader import ClinicalFeatureExtractor
from .checkpoint import ClinicalCheckpointManager
from .validation import (
    ClinicalAuditReport,
    ClinicalDataValidator,
    validate_clinical_representation_output,
)

__version__ = "1.0.0"

__all__ = [
    "ClinicalSchema",
    "get_default_retinal_clinical_schema",
    "ClinicalTransformerConfig",
    "get_default_clinical_config",
    "get_clinical_outputs_dir",
    "ClinicalPreprocessor",
    "PreprocessorState",
    "NumericalFeatureTokenizer",
    "CategoricalFeatureTokenizer",
    "ClinicalFeatureTokenizer",
    "FTTransformerBlock",
    "FTTransformerBackbone",
    "AttentiveTokenPooler",
    "ClinicalRepresentationHead",
    "ClinicalFTTransformerModel",
    "ClinicalTabularDataset",
    "create_clinical_dataloader",
    "patient_level_split",
    "ClinicalFeatureExtractor",
    "ClinicalCheckpointManager",
    "ClinicalAuditReport",
    "ClinicalDataValidator",
    "validate_clinical_representation_output",
]
