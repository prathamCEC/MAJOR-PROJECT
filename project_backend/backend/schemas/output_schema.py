"""
Pydantic Output Response Schemas for API Endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "retinal-multimodal-ai"
    version: str = "1.0.0"
    device: str = "cpu"


class ModelStatusResponse(BaseModel):
    phase4_octa: str = Field(default="loaded", description="Phase 4 Swin OCT-A Backbone status")
    phase4_octb: str = Field(default="loaded", description="Phase 4 Swin OCT-B Backbone status")
    phase4_fundus: str = Field(default="loaded", description="Phase 4 Swin Fundus Backbone status")
    phase5: str = Field(default="loaded", description="Phase 5 Retinal DMRA & Cross-Attention Fusion")
    phase6: str = Field(default="loaded", description="Phase 6 Clinical FT-Transformer")
    phase7: str = Field(default="loaded", description="Phase 7 Retina-Clinical Fusion")
    phase8: str = Field(default="loaded", description="Phase 8 Multi-Task Prediction Network")
    phase9: str = Field(default="available", description="Phase 9 Monte Carlo Dropout Engine")
    phase10: str = Field(default="available", description="Phase 10 Grad-CAM & SHAP Explainability")
    phase11: str = Field(default="available", description="Phase 11 Clinical Report & PDF Generator")
    device: str = "cpu"


class ModalityQualityItem(BaseModel):
    available: bool
    quality_score: Optional[float] = None
    decision: str = "Not available"
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DiseasePredictionItem(BaseModel):
    predicted_class: int
    probability: float
    risk_category: str
    class_label: str


class UncertaintyItem(BaseModel):
    confidence_percent: float
    predictive_variance: float
    predictive_entropy: float
    uncertainty_level: str
    confidence_level: str
    is_elevated_uncertainty: bool
    statement: str


class GradCAMResponseItem(BaseModel):
    status: str
    panel_path: Optional[str] = None
    original_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    panel_url: Optional[str] = None


class ClinicalSHAPItemResponse(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: str


class AnalysisResponse(BaseModel):
    status: str = "success"
    session_id: Optional[str] = None
    report_id: str
    patient_id: str
    timestamp: str
    modalities_processed: List[str]
    image_quality: Dict[str, ModalityQualityItem]
    modality_attribution: Dict[str, float]
    stroke_prediction: DiseasePredictionItem
    stroke_uncertainty: UncertaintyItem
    alzheimer_prediction: DiseasePredictionItem
    alzheimer_uncertainty: UncertaintyItem
    overall_risk_level: str = "LOW"
    explainability: Dict[str, Any]
    clinical_summary: str
    pdf_report_path: str
    pdf_download_url: str
    json_report_path: str
    disclaimer: str
