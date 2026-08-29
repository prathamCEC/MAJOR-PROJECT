"""
Tests for Clinical PDF report generation with ReportLab.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from phase_11_report_generator.config import ReportConfig
from phase_11_report_generator.report_data import (
    ClinicalReportData,
    PatientDemographics,
    ImageQualityItem,
    DiseaseAssessmentItem,
    GradCAMItem,
    ClinicalSHAPItem,
    ExplainabilitySummary,
)
from phase_11_report_generator.pdf_generator import ClinicalPDFReportGenerator


def test_pdf_generation_complete(tmp_path: Path):
    # Create mock Grad-CAM panel image
    panel_img = tmp_path / "mock_panel.png"
    Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)).save(panel_img)

    # Create mock SHAP plot image
    shap_img = tmp_path / "mock_shap.png"
    Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)).save(shap_img)

    data = ClinicalReportData(
        report_id="REP-PDF-001",
        patient_id="PATIENT-PDF-001",
        generated_at="2026-08-29 10:00:00",
        system_version="v1.0.0",
        modalities_available={"octa": True, "octb": True, "fundus": False},
        patient_demographics=PatientDemographics(
            patient_id="PATIENT-PDF-001",
            age_group="O_CD",
            gender="Male (1)",
            education_years="16",
            bmi="26.5",
            hypertension="Positive (1)",
            diabetes_type2="Negative (0)",
        ),
        image_quality={
            "octa": ImageQualityItem(modality="octa", available=True, quality_score=91.0, decision="ACCEPT"),
            "octb": ImageQualityItem(modality="octb", available=True, quality_score=87.5, decision="ACCEPT"),
            "fundus": ImageQualityItem(modality="fundus", available=False, decision="Not available"),
        },
        stroke_assessment=DiseaseAssessmentItem(
            disease_name="Stroke",
            predicted_class=1,
            probability=0.68,
            confidence_percent=94.2,
            uncertainty_level="LOW",
            confidence_level="HIGH",
            predictive_variance=0.003,
            predictive_entropy=0.31,
            risk_category="HIGH RISK",
        ),
        alzheimer_assessment=DiseaseAssessmentItem(
            disease_name="Alzheimer's Disease",
            predicted_class=0,
            probability=0.32,
            confidence_percent=88.5,
            uncertainty_level="LOW",
            confidence_level="HIGH",
            predictive_variance=0.006,
            predictive_entropy=0.28,
            risk_category="LOW RISK",
        ),
        explainability=ExplainabilitySummary(
            stroke_gradcam={"octa": GradCAMItem(modality="octa", status="SUCCESS", panel_path=str(panel_img))},
            stroke_shap_clinical=[ClinicalSHAPItem("HTN", 1, 0.45, "INCREASES_RISK")],
            stroke_shap_plot_path=str(shap_img),
        ),
        clinical_summary_text="The model estimated probability and uncertainty for research evaluation.",
        multimodal_summary_text="Multimodal pathway summary.",
        limitations_text="Research limitations.",
        disclaimer_text="IMPORTANT RESEARCH NOTICE: Not a confirmed clinical diagnosis.",
    )

    out_pdf = tmp_path / "test_report.pdf"
    generator = ClinicalPDFReportGenerator(config=ReportConfig(output_dir=str(tmp_path)))
    pdf_path = generator.generate_pdf(report_data=data, output_filepath=out_pdf)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000  # Verify non-trivial PDF content


def test_pdf_generation_resilient_to_missing_images(tmp_path: Path):
    """Verify PDF generator does not crash if an optional Grad-CAM image is missing on disk."""
    data = ClinicalReportData(
        report_id="REP-PDF-NO-IMG",
        patient_id="PATIENT-NO-IMG",
        generated_at="2026-08-29 10:00:00",
        system_version="v1.0.0",
        modalities_available={"octa": False, "octb": False, "fundus": False},
        patient_demographics=PatientDemographics(patient_id="PATIENT-NO-IMG"),
        image_quality={},
        stroke_assessment=DiseaseAssessmentItem(
            disease_name="Stroke",
            predicted_class=0,
            probability=0.20,
            confidence_percent=90.0,
            uncertainty_level="LOW",
            confidence_level="HIGH",
            predictive_variance=0.002,
            predictive_entropy=0.15,
            risk_category="LOW RISK",
        ),
        alzheimer_assessment=DiseaseAssessmentItem(
            disease_name="Alzheimer's Disease",
            predicted_class=0,
            probability=0.15,
            confidence_percent=92.0,
            uncertainty_level="LOW",
            confidence_level="HIGH",
            predictive_variance=0.001,
            predictive_entropy=0.12,
            risk_category="LOW RISK",
        ),
        explainability=ExplainabilitySummary(
            stroke_gradcam={"octa": GradCAMItem(modality="octa", status="SUCCESS", panel_path=str(tmp_path / "non_existent.png"))}
        ),
        clinical_summary_text="Summary text.",
        multimodal_summary_text="Multimodal summary.",
        limitations_text="Limitations.",
        disclaimer_text="IMPORTANT RESEARCH NOTICE: Research only.",
    )

    out_pdf = tmp_path / "test_no_img_report.pdf"
    generator = ClinicalPDFReportGenerator(config=ReportConfig(output_dir=str(tmp_path)))
    pdf_path = generator.generate_pdf(report_data=data, output_filepath=out_pdf)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 500
