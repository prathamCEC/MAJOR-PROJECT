"""
Command Line Interface for Phase 10 Model Explainability (Grad-CAM + SHAP).

Provides CLI utilities to inspect explainability configurations and generate
multimodal heatmaps and feature attribution reports for individual patient profiles.
"""

import argparse
from pathlib import Path
import sys
import torch

from .config import ExplainabilityConfig, get_default_explainability_config, get_phase10_outputs_dir
from .explainability_engine import MultimodalExplainabilityEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 — Multimodal Explainability Engine (Swin Grad-CAM + SHAP)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print Phase 10 explainability configuration summary")

    # 2. Explain Command
    exp_parser = subparsers.add_parser("explain", help="Generate Grad-CAM heatmaps and SHAP explanations for patient")
    exp_parser.add_argument("--patient-id", type=str, default="DEMO_PATIENT_01", help="Patient Identifier")
    exp_parser.add_argument("--octa", type=str, default=None, help="Path to OCT-A retinal scan")
    exp_parser.add_argument("--octb", type=str, default=None, help="Path to OCT-B retinal scan")
    exp_parser.add_argument("--fundus", type=str, default=None, help="Path to Fundus retinal scan")
    exp_parser.add_argument("--output", type=str, default=None, help="Directory to save explanation figures")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_explainability_config()
        print("\n============================================================")
        print("PHASE 10 — MULTIMODAL EXPLAINABILITY (GRAD-CAM + SHAP)")
        print("============================================================")
        print(f"Grad-CAM Enabled         : {cfg.gradcam_enabled}")
        print(f"Grad-CAM Target Layer    : {cfg.gradcam_target_layer or 'Auto-discovered (Last Swin Stage)'}")
        print(f"Grad-CAM Colormap        : {cfg.gradcam_colormap}")
        print(f"Grad-CAM Alpha Overlay   : {cfg.gradcam_alpha}")
        print(f"SHAP Enabled             : {cfg.shap_enabled}")
        print(f"SHAP Background Samples  : {cfg.shap_background_samples}")
        print(f"Include Phase 9 MC-Unc   : {cfg.include_phase9_uncertainty}")
        print(f"Device                   : {cfg.get_device()}")
        print(f"Default Output Dir       : {cfg.get_output_dir()}")
        print("============================================================")
        print("DISCLAIMER: RESEARCH EXPLANATIONS ONLY — NOT CLINICAL DIAGNOSIS")
        print("============================================================\n")

    elif args.command == "explain":
        cfg = get_default_explainability_config()
        if args.output:
            cfg.output_dir = args.output

        scans = {}
        if args.octa:
            scans["octa"] = args.octa
        if args.octb:
            scans["octb"] = args.octb
        if args.fundus:
            scans["fundus"] = args.fundus

        # Demo clinical record if not specified via file
        demo_record = {
            "ID#": args.patient_id,
            "Old groups": "O_CD",
            "Gender": 1,
            "Education": 14.0,
            "BMI": 27.5,
            "Obese": 0.0,
            "EtOH_ever": 1,
            "EtOH_current": 0,
            "Smoking_ever": 1,
            "Smoking_current": 0,
            "HTN": 1,
            "DM2": 0,
        }

        print("\n============================================================")
        print(f"GENERATING EXPLANATIONS FOR PATIENT: {args.patient_id}")
        print("============================================================")
        engine = MultimodalExplainabilityEngine(config=cfg)
        res = engine.explain_patient(
            patient_id=args.patient_id,
            retinal_scans=scans,
            clinical_record=demo_record,
            save_plots=True,
        )

        st = res["stroke"]
        al = res["alzheimer"]
        print(f"Stroke Prediction      : Class {st['predicted_class']} (Prob: {st['probability']:.4f})")
        print(f"Stroke Confidence      : {st['uncertainty']['confidence_percent']:.2f}% (Entropy: {st['uncertainty']['predictive_entropy']:.4f})")
        print(f"Alzheimer's Prediction : Class {al['predicted_class']} (Prob: {al['probability']:.4f})")
        print(f"Alzheimer's Confidence : {al['uncertainty']['confidence_percent']:.2f}% (Entropy: {al['uncertainty']['predictive_entropy']:.4f})")
        print("------------------------------------------------------------")
        print("Modality Attributions:")
        for k, v in res["modality_attribution"].items():
            print(f"  {k:30s}: {v}%")
        print("------------------------------------------------------------")
        print("Top Clinical Contributors (Stroke):")
        for item in st["shap_clinical"]["summary"][:3]:
            print(f"  {item['feature']:15s} (val={item['value']}): SHAP={item['shap_value']:+.4f} -> {item['direction']}")
        print("------------------------------------------------------------")
        print("Top Clinical Contributors (Alzheimer's):")
        for item in al["shap_clinical"]["summary"][:3]:
            print(f"  {item['feature']:15s} (val={item['value']}): SHAP={item['shap_value']:+.4f} -> {item['direction']}")
        print("============================================================")
        print(f"Saved explanation figures and plots to: {cfg.get_output_dir() / args.patient_id}\n")


if __name__ == "__main__":
    main()
