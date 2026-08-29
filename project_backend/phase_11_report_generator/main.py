"""
Command Line Interface for Phase 11 Clinical-Style Report Generator & PDF Generation.

Provides CLI sub-commands to inspect report configurations and compile PDF + JSON reports.
"""

import argparse
from pathlib import Path
import sys

from .config import ReportConfig, get_default_report_config, get_default_reports_dir
from .report_generator import ClinicalReportGenerator
from .pipeline import EndToEndReportPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 11 — Clinical-Style Assessment Report Generator (PDF + JSON)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print Phase 11 report configuration summary")

    # 2. Generate Command
    gen_parser = subparsers.add_parser("generate", help="Generate PDF & JSON clinical report for a patient")
    gen_parser.add_argument("--patient-id", type=str, default="DEMO_PATIENT_01", help="Patient Identifier")
    gen_parser.add_argument("--octa", type=str, default=None, help="Path to OCT-A scan")
    gen_parser.add_argument("--octb", type=str, default=None, help="Path to OCT-B scan")
    gen_parser.add_argument("--fundus", type=str, default=None, help="Path to Fundus scan")
    gen_parser.add_argument("--output-pdf", type=str, default=None, help="Custom output PDF path")
    gen_parser.add_argument("--output-json", type=str, default=None, help="Custom output JSON path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_report_config()
        print("\n============================================================")
        print("PHASE 11 — CLINICAL-STYLE ASSESSMENT REPORT GENERATOR")
        print("============================================================")
        print(f"System Title             : {cfg.system_title}")
        print(f"Document Version         : {cfg.document_version}")
        print(f"Low Risk Threshold       : < {cfg.low_risk_threshold}")
        print(f"Moderate Risk Threshold  : < {cfg.moderate_risk_threshold}")
        print(f"High Risk Threshold      : >= {cfg.moderate_risk_threshold}")
        print(f"High Confidence Cutoff   : >= {cfg.high_confidence_threshold}%")
        print(f"Default Reports Dir      : {cfg.get_output_path()}")
        print(f"PDF Output Dir           : {cfg.get_pdf_dir()}")
        print(f"JSON Output Dir          : {cfg.get_json_dir()}")
        print("============================================================")
        print("MANDATORY DISCLAIMER:")
        print(f"\"{cfg.disclaimer}\"")
        print("============================================================\n")

    elif args.command == "generate":
        cfg = get_default_report_config()
        scans = {}
        if args.octa:
            scans["octa"] = args.octa
        if args.octb:
            scans["octb"] = args.octb
        if args.fundus:
            scans["fundus"] = args.fundus

        demo_record = {
            "ID#": args.patient_id,
            "Old groups": "O_CD",
            "Gender": 1,
            "Education": 16.0,
            "BMI": 26.8,
            "Obese": 0.0,
            "EtOH_ever": 1,
            "EtOH_current": 0,
            "Smoking_ever": 1,
            "Smoking_current": 0,
            "HTN": 1,
            "DM2": 0,
        }

        print("\n============================================================")
        print(f"GENERATING CLINICAL REPORT FOR PATIENT: {args.patient_id}")
        print("============================================================")

        pipeline = EndToEndReportPipeline(report_config=cfg)
        result = pipeline.process_patient_and_generate_report(
            patient_id=args.patient_id,
            retinal_scans=scans,
            clinical_record=demo_record,
            pdf_path=args.output_pdf,
            json_path=args.output_json,
        )

        st = result["report_data"].stroke_assessment
        al = result["report_data"].alzheimer_assessment

        print(f"Report ID           : {result['report_id']}")
        print(f"Stroke Assessment   : {st.risk_category} (Prob: {st.probability:.4f}, Conf: {st.confidence_percent:.2f}%)")
        print(f"Alzheimer's Assess  : {al.risk_category} (Prob: {al.probability:.4f}, Conf: {al.confidence_percent:.2f}%)")
        print("------------------------------------------------------------")
        print(f"Saved PDF Report to : {result['pdf_path']}")
        print(f"Saved JSON Report to: {result['json_path']}")
        if result["csv_path"]:
            print(f"Appended to CSV log : {result['csv_path']}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
