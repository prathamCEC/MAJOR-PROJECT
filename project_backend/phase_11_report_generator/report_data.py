"""
Structured Data Containers and Schema Validation for Clinical-Style Reports.

Encapsulates patient demographics, image quality metrics, Phase 8 multi-task predictions,
Phase 9 uncertainty statistics, and Phase 10 Grad-CAM/SHAP explainability attributions.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class PatientDemographics:
    """Patient clinical metadata and demographics."""
    patient_id: str
    age_group: str = "Not provided"
    gender: str = "Not provided"
    education_years: str = "Not provided"
    bmi: str = "Not provided"
    hypertension: str = "Not provided"
    diabetes_type2: str = "Not provided"
    smoking_status: str = "Not provided"
    alcohol_status: str = "Not provided"
    raw_clinical_record: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageQualityItem:
    """Image Quality Assessment (Phase 3) details for a single modality."""
    modality: str
    available: bool = False
    quality_score: Optional[float] = None
    decision: str = "Not available"  # ACCEPT, REJECT, or Not available
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiseaseAssessmentItem:
    """Disease-specific prediction, confidence, uncertainty, and risk categorization."""
    disease_name: str  # "Stroke" or "Alzheimer's Disease"
    predicted_class: int  # 0 or 1
    probability: float  # [0.0, 1.0]
    confidence_percent: float  # [0.0, 100.0]
    uncertainty_level: str  # "LOW", "MODERATE", "ELEVATED"
    confidence_level: str  # "HIGH", "MODERATE", "LOW"
    predictive_variance: float
    predictive_entropy: float
    risk_category: str  # "LOW RISK", "MODERATE RISK", "HIGH RISK"
    is_elevated_uncertainty: bool = False


@dataclass
class GradCAMItem:
    """Grad-CAM spatial visualization paths for a specific modality and target disease."""
    modality: str
    status: str = "UNAVAILABLE"  # "SUCCESS", "UNAVAILABLE", "ERROR"
    panel_path: Optional[str] = None
    original_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    caption: str = "Grad-CAM visualization of salient retinal regions contributing to model prediction."


@dataclass
class ClinicalSHAPItem:
    """Clinical feature Shapley attribution."""
    feature_name: str
    patient_value: Any
    shap_value: float
    direction: str  # "INCREASES_RISK", "DECREASES_RISK", "NEUTRAL"


@dataclass
class ExplainabilitySummary:
    """Multimodal explainability bundle from Phase 10."""
    modality_attributions: Dict[str, float] = field(default_factory=dict)
    stroke_gradcam: Dict[str, GradCAMItem] = field(default_factory=dict)
    alzheimer_gradcam: Dict[str, GradCAMItem] = field(default_factory=dict)
    stroke_shap_clinical: List[ClinicalSHAPItem] = field(default_factory=list)
    alzheimer_shap_clinical: List[ClinicalSHAPItem] = field(default_factory=list)
    stroke_shap_plot_path: Optional[str] = None
    alzheimer_shap_plot_path: Optional[str] = None


@dataclass
class ClinicalReportData:
    """
    Unified Data Model for Phase 11 Multimodal Assessment Report.
    """
    report_id: str
    patient_id: str
    generated_at: str
    system_version: str
    modalities_available: Dict[str, bool]
    patient_demographics: PatientDemographics
    image_quality: Dict[str, ImageQualityItem]
    stroke_assessment: DiseaseAssessmentItem
    alzheimer_assessment: DiseaseAssessmentItem
    explainability: ExplainabilitySummary
    clinical_summary_text: str
    multimodal_summary_text: str
    limitations_text: str
    disclaimer_text: str

    def validate(self) -> List[str]:
        """
        Validate data integrity, mathematical bounds, and presence of required fields.
        Returns a list of error messages (empty if valid).
        """
        errors = []

        # Validate Report & Patient IDs
        if not self.report_id:
            errors.append("Missing report_id.")
        if not self.patient_id:
            errors.append("Missing patient_id.")

        # Validate Stroke Assessment
        st = self.stroke_assessment
        if not isinstance(st.predicted_class, int) or st.predicted_class not in (0, 1):
            errors.append(f"Invalid Stroke predicted_class: {st.predicted_class} (must be 0 or 1).")
        if math.isnan(st.probability) or math.isinf(st.probability) or not (0.0 <= st.probability <= 1.0):
            errors.append(f"Invalid Stroke probability: {st.probability} (must be in [0, 1]).")
        if math.isnan(st.confidence_percent) or not (0.0 <= st.confidence_percent <= 100.0):
            errors.append(f"Invalid Stroke confidence_percent: {st.confidence_percent} (must be in [0, 100]).")
        if math.isnan(st.predictive_variance) or st.predictive_variance < 0.0:
            errors.append(f"Invalid Stroke predictive_variance: {st.predictive_variance}.")

        # Validate Alzheimer's Assessment
        al = self.alzheimer_assessment
        if not isinstance(al.predicted_class, int) or al.predicted_class not in (0, 1):
            errors.append(f"Invalid Alzheimer predicted_class: {al.predicted_class} (must be 0 or 1).")
        if math.isnan(al.probability) or math.isinf(al.probability) or not (0.0 <= al.probability <= 1.0):
            errors.append(f"Invalid Alzheimer probability: {al.probability} (must be in [0, 1]).")
        if math.isnan(al.confidence_percent) or not (0.0 <= al.confidence_percent <= 100.0):
            errors.append(f"Invalid Alzheimer confidence_percent: {al.confidence_percent} (must be in [0, 100]).")
        if math.isnan(al.predictive_variance) or al.predictive_variance < 0.0:
            errors.append(f"Invalid Alzheimer predictive_variance: {al.predictive_variance}.")

        # Validate Disclaimer presence
        if not self.disclaimer_text or "RESEARCH" not in self.disclaimer_text.upper():
            errors.append("Mandatory research disclaimer text is missing or invalid.")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert data model into a clean JSON-serializable dictionary."""
        return asdict(self)
