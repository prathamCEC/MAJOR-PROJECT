"""
Phase 10: Model Explainability Using Grad-CAM + SHAP.

Provides visual and game-theoretic attributions for Stroke and Alzheimer's disease predictions:
- Swin Transformer Grad-CAM for OCT-A, OCT-B, and Fundus retinal scans
- Clinical Feature SHAP for patient health variables
- Multimodal attribution across retinal and tabular pathways
- Integrated Phase 9 Monte Carlo Dropout uncertainty metrics
"""

from .config import (
    ExplainabilityConfig,
    get_default_explainability_config,
    get_phase10_outputs_dir,
)
from .swin_gradcam import SwinGradCAM
from .shap_explainer import MultimodalSHAPExplainer
from .visualization import (
    overlay_cam_on_image,
    save_gradcam_panel,
    save_shap_bar_chart,
)
from .explainability_engine import MultimodalExplainabilityEngine
from .pipeline import EndToEndExplainabilityPipeline

__version__ = "1.0.0"

__all__ = [
    "ExplainabilityConfig",
    "get_default_explainability_config",
    "get_phase10_outputs_dir",
    "SwinGradCAM",
    "MultimodalSHAPExplainer",
    "overlay_cam_on_image",
    "save_gradcam_panel",
    "save_shap_bar_chart",
    "MultimodalExplainabilityEngine",
    "EndToEndExplainabilityPipeline",
]
