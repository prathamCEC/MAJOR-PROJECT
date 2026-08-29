"""
JSON & CSV Export Utilities for Phase 11 Reports.

Exports full machine-readable schema for future API/web frontend ingestion
and appends summary records to aggregate research CSV logs.
"""

import csv
import json
from pathlib import Path
from typing import Optional, Union

from .config import ReportConfig, get_default_report_config
from .report_data import ClinicalReportData


class JSONReportExporter:
    """
    Exports ClinicalReportData to structured JSON and CSV formats.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or get_default_report_config()

    def export_json(
        self,
        report_data: ClinicalReportData,
        output_filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save machine-readable JSON representation of report.

        Args:
            report_data: Validated ClinicalReportData instance.
            output_filepath: Destination JSON file path (optional).

        Returns:
            Resolved Path of the saved JSON file.
        """
        if output_filepath:
            json_path = Path(output_filepath).resolve()
        else:
            json_dir = self.config.get_json_dir()
            json_path = json_dir / f"report_{report_data.patient_id}_{report_data.report_id}.json"

        json_path.parent.mkdir(parents=True, exist_ok=True)

        data_dict = report_data.to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        return json_path

    def append_summary_csv(
        self,
        report_data: ClinicalReportData,
        csv_filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Append high-level assessment summary row to aggregate CSV table.

        Args:
            report_data: Validated ClinicalReportData instance.
            csv_filepath: Destination CSV file path (optional).

        Returns:
            Resolved Path of the CSV file.
        """
        if csv_filepath:
            csv_path = Path(csv_filepath).resolve()
        else:
            csv_dir = self.config.get_summary_csv_dir()
            csv_path = csv_dir / "reports_summary.csv"

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()

        st = report_data.stroke_assessment
        al = report_data.alzheimer_assessment

        fieldnames = [
            "report_id",
            "patient_id",
            "timestamp",
            "stroke_pred_class",
            "stroke_probability",
            "stroke_confidence_percent",
            "stroke_uncertainty_level",
            "stroke_risk_category",
            "alzheimer_pred_class",
            "alzheimer_probability",
            "alzheimer_confidence_percent",
            "alzheimer_uncertainty_level",
            "alzheimer_risk_category",
        ]

        row = {
            "report_id": report_data.report_id,
            "patient_id": report_data.patient_id,
            "timestamp": report_data.generated_at,
            "stroke_pred_class": st.predicted_class,
            "stroke_probability": f"{st.probability:.4f}",
            "stroke_confidence_percent": f"{st.confidence_percent:.2f}",
            "stroke_uncertainty_level": st.uncertainty_level,
            "stroke_risk_category": st.risk_category,
            "alzheimer_pred_class": al.predicted_class,
            "alzheimer_probability": f"{al.probability:.4f}",
            "alzheimer_confidence_percent": f"{al.confidence_percent:.2f}",
            "alzheimer_uncertainty_level": al.uncertainty_level,
            "alzheimer_risk_category": al.risk_category,
        }

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return csv_path
