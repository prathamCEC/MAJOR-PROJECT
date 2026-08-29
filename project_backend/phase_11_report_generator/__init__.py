"""
Phase 11: Clinical-Style Report Generator & PDF Generation.

Delivers multi-page, formatted research assessment reports:
- Dual-target Disease Categorization (Stroke + Alzheimer's Disease)
- Phase 9 Uncertainty & Confidence Metrics
- Phase 10 Retinal Swin Grad-CAM Heatmaps & Clinical SHAP Values
- Multi-page PDF Generator with Dynamic Pagination
- Machine-readable JSON Exporter & Aggregate CSV Summary Logging
"""

from .config import (
    ReportConfig,
    get_default_report_config,
    get_default_reports_dir,
)
from .report_data import (
    ClinicalReportData,
    PatientDemographics,
    ImageQualityItem,
    DiseaseAssessmentItem,
    GradCAMItem,
    ClinicalSHAPItem,
    ExplainabilitySummary,
)
from .risk_calculator import (
    calculate_risk_category,
    calculate_confidence_category,
    get_uncertainty_statement,
)
from .summary_builder import (
    build_clinical_narrative_summary,
    build_multimodal_structural_summary,
    get_default_limitations_text,
)
from .pdf_generator import ClinicalPDFReportGenerator
from .json_generator import JSONReportExporter
from .report_generator import ClinicalReportGenerator
from .pipeline import EndToEndReportPipeline

__version__ = "1.0.0"

__all__ = [
    "ReportConfig",
    "get_default_report_config",
    "get_default_reports_dir",
    "ClinicalReportData",
    "PatientDemographics",
    "ImageQualityItem",
    "DiseaseAssessmentItem",
    "GradCAMItem",
    "ClinicalSHAPItem",
    "ExplainabilitySummary",
    "calculate_risk_category",
    "calculate_confidence_category",
    "get_uncertainty_statement",
    "build_clinical_narrative_summary",
    "build_multimodal_structural_summary",
    "get_default_limitations_text",
    "ClinicalPDFReportGenerator",
    "JSONReportExporter",
    "ClinicalReportGenerator",
    "EndToEndReportPipeline",
]
