"""
Tests for ClinicalReportData schema validation and mathematical bounds.
"""

import math
import pytest

from phase_11_report_generator.report_data import (
    ClinicalReportData,
    PatientDemographics,
    ImageQualityItem,
    DiseaseAssessmentItem,
    ExplainabilitySummary,
)


def create_valid_report_data() -> ClinicalReportData:
    return ClinicalReportData(
        report_id="REP-TEST-001",
        patient_id="PATIENT-001",
        generated_at="2026-08-29 10:00:00",
        system_version="v1.0.0",
        modalities_available={"octa": True, "octb": False, "fundus": False},
        patient_demographics=PatientDemographics(patient_id="PATIENT-001"),
        image_quality={"octa": ImageQualityItem(modality="octa", available=True, quality_score=88.5, decision="ACCEPT")},
        stroke_assessment=DiseaseAssessmentItem(
            disease_name="Stroke",
            predicted_class=1,
            probability=0.72,
            confidence_percent=92.5,
            uncertainty_level="LOW",
            confidence_level="HIGH",
            predictive_variance=0.005,
            predictive_entropy=0.35,
            risk_category="HIGH RISK",
        ),
        alzheimer_assessment=DiseaseAssessmentItem(
            disease_name="Alzheimer's Disease",
            predicted_class=0,
            probability=0.24,
            confidence_percent=89.0,
            uncertainty_level="LOW",
            confidence_level="HIGH",
            predictive_variance=0.008,
            predictive_entropy=0.22,
            risk_category="LOW RISK",
        ),
        explainability=ExplainabilitySummary(),
        clinical_summary_text="Valid test clinical summary.",
        multimodal_summary_text="Valid multimodal summary.",
        limitations_text="Limitations.",
        disclaimer_text="IMPORTANT RESEARCH NOTICE: For experimental research only.",
    )


def test_valid_report_data_passes_validation():
    data = create_valid_report_data()
    errors = data.validate()
    assert len(errors) == 0, f"Expected no validation errors, got: {errors}"


def test_invalid_probability_fails_validation():
    data = create_valid_report_data()
    data.stroke_assessment.probability = 1.5  # Out of [0, 1]
    errors = data.validate()
    assert any("probability" in err for err in errors)

    data.stroke_assessment.probability = float("nan")
    errors = data.validate()
    assert any("probability" in err for err in errors)


def test_invalid_confidence_fails_validation():
    data = create_valid_report_data()
    data.alzheimer_assessment.confidence_percent = -5.0
    errors = data.validate()
    assert any("confidence" in err for err in errors)


def test_missing_disclaimer_fails_validation():
    data = create_valid_report_data()
    data.disclaimer_text = ""
    errors = data.validate()
    assert any("disclaimer" in err for err in errors)
