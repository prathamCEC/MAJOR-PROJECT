"""
Master End-to-End Multimodal Retinal AI System Runner.

Orchestrates the entire Phase 2 -> Phase 11 pipeline in a single unified execution:
Phase 2 (Preprocessing) -> Phase 3 (Quality Assessment) -> Phase 4 (Swin Backbones) ->
Phase 5 (Retinal Fusion URR) -> Phase 6 (Clinical FT-Transformer CR) ->
Phase 7 (Retina-Clinical Cross-Attention UPR) -> Phase 8 (Multi-Task Disease Predictions) ->
Phase 9 (Monte Carlo Dropout Uncertainty) -> Phase 10 (Swin Grad-CAM + Clinical SHAP) ->
Phase 11 (Clinical-Style PDF & JSON Report Generation).
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import subprocess
import torch

from phase_11_report_generator.config import get_default_report_config
from phase_11_report_generator.pipeline import EndToEndReportPipeline


def run_full_pipeline(
    patient_id: str = "PATIENT_MASTER_01",
    octa_path: str = None,
    octb_path: str = None,
    fundus_path: str = None,
    clinical_file: str = None,
    run_tests: bool = False,
) -> None:
    backend_root = Path(__file__).resolve().parent

    print("\n" + "=" * 70)
    print("AI-BASED MULTIMODAL RETINAL ANALYSIS SYSTEM — MASTER PIPELINE")
    print("Stroke & Alzheimer's Disease Multimodal Detection & Reporting")
    print("=" * 70)
    print(f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Patient Identifier : {patient_id}")
    print(f"PyTorch Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    # 1. Resolve Available Retinal Scans
    scans = {}
    default_octa = backend_root / "datasets" / "approved" / "octa" / "octa_sample_1_processed.png"
    default_octb = backend_root / "datasets" / "approved" / "octb" / "octb_sample_1_processed.png"
    default_fundus = backend_root / "datasets" / "approved" / "fundus" / "fundus_sample_1_processed.png"

    p_octa = Path(octa_path).resolve() if octa_path else default_octa
    p_octb = Path(octb_path).resolve() if octb_path else default_octb
    p_fundus = Path(fundus_path).resolve() if fundus_path else default_fundus

    if p_octa.exists():
        scans["octa"] = p_octa
    if p_octb.exists():
        scans["octb"] = p_octb
    if p_fundus.exists():
        scans["fundus"] = p_fundus

    print(f"\n[1/4] Retinal Scans Loaded:")
    for mod, path in scans.items():
        print(f"  * {mod.upper():6s} : {path.name}")
    if not scans:
        print("  * (No local image files found; proceeding with Tabular Clinical profile only)")

    # 2. Patient Tabular Clinical Variables
    clinical_record = {
        "ID#": patient_id,
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 16.0,
        "BMI": 27.4,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 0,
    }
    print(f"\n[2/4] Clinical Tabular Variables Loaded:")
    print(f"  * Demographics: Gender=Male (1), Education=16 yrs, BMI=27.4")
    print(f"  * Risk Factors: HTN=Positive (1), DM2=Negative (0), Smoking=History (1)")

    # 3. Execute End-to-End Orchestrator (Phases 4 through 11)
    print(f"\n[3/4] Executing Multimodal Deep Learning & Explainability Engine...")
    print("  -> Phase 4 : Swin Transformer Backbone Encoders (OCT-A, OCT-B, Fundus)")
    print("  -> Phase 5 : DMRA Dynamic Reliability Attention & URR Retinal Fusion")
    print("  -> Phase 6 : FT-Transformer Clinical Feature Tokenizer & CR Extraction")
    print("  -> Phase 7 : Bidirectional Retina-Clinical Cross-Attention Fusion (UPR)")
    print("  -> Phase 8 : Multi-Task Dual Disease Prediction Network (Stroke + AD)")
    print("  -> Phase 9 : Monte Carlo Dropout Uncertainty & Confidence Estimation")
    print("  -> Phase 10: Swin Grad-CAM Heatmaps & Clinical SHAP Attributions")
    print("  -> Phase 11: Clinical-Style PDF & JSON Report Compilation")

    pipeline = EndToEndReportPipeline()
    result = pipeline.process_patient_and_generate_report(
        patient_id=patient_id,
        retinal_scans=scans,
        clinical_record=clinical_record,
    )

    report_data = result["report_data"]
    st = report_data.stroke_assessment
    al = report_data.alzheimer_assessment

    print(f"\n[4/4] Assessment Results & Generated Artifacts:")
    print("-" * 70)
    print(f"Report ID           : {result['report_id']}")
    print(f"Stroke Assessment   : {st.risk_category} (Prob: {st.probability:.4f}, Conf: {st.confidence_percent:.2f}%, Var: {st.predictive_variance:.4f})")
    print(f"Alzheimer's Assess  : {al.risk_category} (Prob: {al.probability:.4f}, Conf: {al.confidence_percent:.2f}%, Var: {al.predictive_variance:.4f})")
    print("-" * 70)
    print(f"[PDF]  PDF Report Saved : {result['pdf_path']}")
    print(f"[JSON] JSON Data Saved  : {result['json_path']}")
    if result.get("csv_path"):
        print(f"[CSV]  Summary CSV Log  : {result['csv_path']}")
    print("=" * 70)
    print("DISCLAIMER: AI Research Prototype Output — For Decision Support Only")
    print("=" * 70)

    # 4. Optional Pytest Verification
    if run_tests:
        print("\n[Optional] Running Automated Test Suite (Phases 2-11)...")
        subprocess.run(["python", "-m", "pytest", "-v"], cwd=str(backend_root))


def main():
    parser = argparse.ArgumentParser(
        description="Master Runner for Multimodal Retinal Disease Detection System"
    )
    parser.add_argument("--patient-id", type=str, default="PATIENT_MASTER_01", help="Patient Identifier")
    parser.add_argument("--octa", type=str, default=None, help="Path to OCT-A retinal image")
    parser.add_argument("--octb", type=str, default=None, help="Path to OCT-B retinal image")
    parser.add_argument("--fundus", type=str, default=None, help="Path to Fundus retinal image")
    parser.add_argument("--test", action="store_true", help="Run full 205-test automated validation suite")

    args = parser.parse_args()
    run_full_pipeline(
        patient_id=args.patient_id,
        octa_path=args.octa,
        octb_path=args.octb,
        fundus_path=args.fundus,
        run_tests=args.test,
    )


if __name__ == "__main__":
    main()
