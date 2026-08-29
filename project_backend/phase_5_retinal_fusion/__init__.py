"""
Phase 5: Dynamic Modality Reliability Attention (DMRA) & Unified Retinal Representation (URR).

Multimodal fusion architecture integrating OCT-A, OCT-B, and Fundus representations
into a unified, robust retinal vector for downstream clinical reasoning (Phase 6 / Phase 7).
"""

from .config import FusionConfig, get_default_fusion_config, get_fusion_outputs_dir
from .modality_projection import SingleModalityProjection, MultiModalityProjection
from .reliability_attention import (
    SingleModalityReliabilityScorer,
    DynamicModalityReliabilityAttention,
)
from .cross_attention import (
    MultiHeadCrossAttentionBlock,
    RetinalCrossAttentionFusion,
)
from .urr import AttentionPoolingHead, UnifiedRetinalRepresentationHead
from .fusion_model import RetinalMultimodalFusionModel
from .feature_loader import Phase4FeatureExtractor
from .validation import (
    validate_input_features,
    validate_modality_mask,
    validate_urr_output,
)

__version__ = "1.0.0"

__all__ = [
    "FusionConfig",
    "get_default_fusion_config",
    "get_fusion_outputs_dir",
    "SingleModalityProjection",
    "MultiModalityProjection",
    "SingleModalityReliabilityScorer",
    "DynamicModalityReliabilityAttention",
    "MultiHeadCrossAttentionBlock",
    "RetinalCrossAttentionFusion",
    "AttentionPoolingHead",
    "UnifiedRetinalRepresentationHead",
    "RetinalMultimodalFusionModel",
    "Phase4FeatureExtractor",
    "validate_input_features",
    "validate_modality_mask",
    "validate_urr_output",
]
