"""
Phase 7: Retina-Clinical Cross-Attention Fusion & Unified Patient Representation (UPR).

Combines Unified Retinal Representation (Phase 5 URR) and Clinical Representation (Phase 6 CR)
using bidirectional cross-attention and gated multimodal fusion.
"""

from .config import (
    RetinaClinicalConfig,
    get_default_retina_clinical_config,
    get_phase7_outputs_dir,
)
from .projection import (
    RepresentationProjectionLayer,
    RetinaClinicalProjection,
)
from .cross_attention import (
    CrossAttentionBlock,
    BidirectionalRetinaClinicalBlock,
    BidirectionalRetinaClinicalTransformer,
)
from .pooling import (
    AttentiveSequencePooler,
    MultimodalTokenPooler,
)
from .fusion import GatedMultimodalFusion
from .fusion_model import RetinaClinicalFusionModel
from .validation import (
    validate_fusion_inputs,
    validate_upr_output,
)
from .feature_loader import PatientMultimodalPipeline
from .checkpoint import Phase7CheckpointManager

__version__ = "1.0.0"

__all__ = [
    "RetinaClinicalConfig",
    "get_default_retina_clinical_config",
    "get_phase7_outputs_dir",
    "RepresentationProjectionLayer",
    "RetinaClinicalProjection",
    "CrossAttentionBlock",
    "BidirectionalRetinaClinicalBlock",
    "BidirectionalRetinaClinicalTransformer",
    "AttentiveSequencePooler",
    "MultimodalTokenPooler",
    "GatedMultimodalFusion",
    "RetinaClinicalFusionModel",
    "validate_fusion_inputs",
    "validate_upr_output",
    "PatientMultimodalPipeline",
    "Phase7CheckpointManager",
]
