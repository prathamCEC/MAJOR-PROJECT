"""
Test PDF Report Generator to strictly enforce:
1. Grad-CAM / heatmap images MUST NOT be embedded in the PDF.
2. The PDF contains patient info, analysis results, uncertainty, and disclaimers.
3. Does NOT describe results as clinically confirmed diagnosis.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

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
from phase_11_report_generator.config import ReportConfig


def test_pdf_report_strictly_excludes_gradcam_heatmaps(tmp_path: Path):
    """Verify generated PDF document contains no heatmap images and has proper disclaimers."""
    pdf_out = tmp_path / "test_clinical_report.pdf"

    # Create dummy panel image
    panel_img = tmp_path / "mock_panel.png"
    Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)).save(panel_img)

    report_data = ClinicalReportData(
        report_id="REP_TEST_999",
        patient_id="PAT_TEST_NO_HEATMAP",
        generated_at="2026-09-03 12:00:00",
        system_version="v1.0.0",
        modalities_available={"octa": True, "octb": False, "fundus": False},
        patient_demographics=PatientDemographics(
            patient_id="PAT_TEST_NO_HEATMAP",
            age_group="O_CD",
            gender="Male",
            education_years="16.0",
            bmi="26.5",
            hypertension="Positive (1)",
            diabetes_type2="Negative (0)",
        ),
        image_quality={
            "octa": ImageQualityItem(modality="octa", available=True, quality_score=91.0, decision="ACCEPT"),
            "octb": ImageQualityItem(modality="octb", available=False, decision="Not available"),
            "fundus": ImageQualityItem(modality="fundus", available=False, decision="Not available"),
        },
        stroke_assessment=DiseaseAssessmentItem(
            disease_name="Stroke",
            predicted_class=0,
            probability=0.22,
            risk_category="LOW RISK",
            confidence_percent=88.5,
            confidence_level="HIGH",
            predictive_variance=0.012,
            uncertainty_level="LOW",
            predictive_entropy=0.34,
        ),
        alzheimer_assessment=DiseaseAssessmentItem(
            disease_name="Alzheimer's Disease",
            predicted_class=1,
            probability=0.74,
            risk_category="HIGH RISK",
            confidence_percent=82.1,
            confidence_level="HIGH",
            predictive_variance=0.035,
            uncertainty_level="MODERATE",
            predictive_entropy=0.58,
        ),
        explainability=ExplainabilitySummary(
            stroke_gradcam={"octa": GradCAMItem(modality="octa", status="SUCCESS", panel_path=str(panel_img))},
            alzheimer_gradcam={"octa": GradCAMItem(modality="octa", status="SUCCESS", panel_path=str(panel_img))},
            stroke_shap_clinical=[ClinicalSHAPItem(feature_name="HTN", patient_value=1.0, shap_value=0.15, direction="INCREASES_RISK")],
            alzheimer_shap_clinical=[ClinicalSHAPItem(feature_name="Age_group", patient_value="O_CD", shap_value=0.25, direction="INCREASES_RISK")],
        ),
        clinical_summary_text="Multimodal evaluation suggests elevated Alzheimer's disease risk indicators.",
        multimodal_summary_text="OCT-A imaging accepted with high technical quality score.",
        limitations_text="Academic research model; requires secondary clinical diagnostic validation.",
        disclaimer_text="RESEARCH DECISION SUPPORT ONLY. Not for independent clinical diagnosis.",
    )

    generator = ClinicalPDFReportGenerator()
    saved_path = generator.generate_pdf(report_data, output_filepath=pdf_out)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000

    # Parse with PdfReader to inspect contents
    reader = PdfReader(str(saved_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # 1. Verify Patient and Analysis Information are present
    assert "PAT_TEST_NO_HEATMAP" in full_text
    assert "REP_TEST_999" in full_text
    assert "Stroke" in full_text
    assert "Alzheimer" in full_text

    # 2. Verify Disclaimers are present
    assert "RESEARCH DECISION SUPPORT" in full_text or "disclaimer" in full_text.lower()
    assert "confirmed clinical diagnosis" not in full_text.lower()

    # 3. CRITICAL: Verify that the document does NOT contain embedded Grad-CAM heatmap images
    for page in reader.pages:
        for img_name in page.images.keys():
            assert "mock_panel" not in img_name
            assert "gradcam" not in img_name.lower()
