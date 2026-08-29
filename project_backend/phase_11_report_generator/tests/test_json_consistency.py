"""
Tests verifying exact numerical and structural consistency between Report Data and JSON Export.
"""

import json
from pathlib import Path
import pytest

from phase_11_report_generator.config import ReportConfig
from phase_11_report_generator.json_generator import JSONReportExporter
from phase_11_report_generator.tests.test_report_data import create_valid_report_data


def test_json_export_and_value_consistency(tmp_path: Path):
    data = create_valid_report_data()
    exporter = JSONReportExporter(config=ReportConfig(output_dir=str(tmp_path)))

    out_json = tmp_path / "test_report.json"
    json_path = exporter.export_json(report_data=data, output_filepath=out_json)

    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Verify key attributes match exactly
    assert loaded["report_id"] == data.report_id
    assert loaded["patient_id"] == data.patient_id

    # Verify Stroke values
    assert loaded["stroke_assessment"]["probability"] == data.stroke_assessment.probability
    assert loaded["stroke_assessment"]["confidence_percent"] == data.stroke_assessment.confidence_percent
    assert loaded["stroke_assessment"]["risk_category"] == data.stroke_assessment.risk_category

    # Verify Alzheimer's values
    assert loaded["alzheimer_assessment"]["probability"] == data.alzheimer_assessment.probability
    assert loaded["alzheimer_assessment"]["confidence_percent"] == data.alzheimer_assessment.confidence_percent
    assert loaded["alzheimer_assessment"]["risk_category"] == data.alzheimer_assessment.risk_category


def test_csv_summary_export(tmp_path: Path):
    data = create_valid_report_data()
    exporter = JSONReportExporter(config=ReportConfig(output_dir=str(tmp_path)))

    out_csv = tmp_path / "summary.csv"
    csv_path = exporter.append_summary_csv(report_data=data, csv_filepath=out_csv)

    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert data.patient_id in content
    assert f"{data.stroke_assessment.probability:.4f}" in content
