"""
Command Line Interface for Phase 6 Clinical FT-Transformer.

Unified CLI supporting clinical schema auditing, preprocessing, and representation extraction.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

from .config import ClinicalTransformerConfig, get_default_clinical_config, get_clinical_outputs_dir
from .schema import ClinicalSchema, get_default_retinal_clinical_schema
from .validation import ClinicalDataValidator
from .feature_loader import ClinicalFeatureExtractor
from .clinical_model import ClinicalFTTransformerModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 — FT-Transformer for Structured Clinical Data & Clinical Representation (CR)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print FT-Transformer architecture and schema summary")

    # 2. Validate Command
    val_parser = subparsers.add_parser("validate", help="Audit and validate a clinical dataset file")
    val_parser.add_argument("--data", type=str, required=True, help="Path to clinical dataset (.xlsx or .csv)")

    # 3. Extract Command
    ext_parser = subparsers.add_parser("extract", help="Extract Clinical Representations from tabular data")
    ext_parser.add_argument("--data", type=str, required=True, help="Path to clinical dataset (.xlsx or .csv)")
    ext_parser.add_argument("--output", type=str, default=None, help="Output path for saved representation .pt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_clinical_config()
        schema = cfg.schema
        print("\n============================================================")
        print("PHASE 6 — CLINICAL FT-TRANSFORMER ARCHITECTURE & SCHEMA")
        print("============================================================")
        print(f"Numerical Features     : {schema.numerical_features}")
        print(f"Categorical Features   : {schema.categorical_features}")
        print(f"Binary Features        : {schema.binary_features}")
        print(f"Patient ID Column      : {schema.patient_id_column}")
        print(f"Feature Token Embed Dim: {cfg.embed_dim}")
        print(f"Self-Attention Heads   : {cfg.num_heads}")
        print(f"Transformer Depth      : {cfg.num_layers} layers")
        print(f"Feed-Forward Dim       : {cfg.ffn_dim}")
        print(f"Representation Dim (CR): {cfg.clinical_representation_dim}")
        print(f"Pooling Strategy       : {cfg.pooling_strategy.upper()}")
        print(f"Device                 : {cfg.get_device()}")
        print("============================================================\n")

    elif args.command == "validate":
        data_path = Path(args.data).resolve()
        if not data_path.exists():
            print(f"Error: Clinical dataset file not found: {data_path}")
            sys.exit(1)

        df = pd.read_excel(data_path) if data_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(data_path)
        schema = get_default_retinal_clinical_schema()
        validator = ClinicalDataValidator(schema=schema)
        report = validator.audit_dataframe(df)

        print("\n============================================================")
        print("PHASE 6 CLINICAL DATA AUDIT REPORT")
        print("============================================================")
        print(f"Total Patient Records  : {report.total_records}")
        print(f"Unique Patient IDs     : {report.unique_patients}")
        print(f"Duplicate Patient IDs  : {report.has_duplicate_patients}")
        print(f"Schema Conformance     : {'[PASS]' if report.is_valid else '[FAIL]'}")
        if report.warnings:
            print("\nWarnings:")
            for w in report.warnings:
                print(f"  [!] {w}")
        if report.errors:
            print("\nErrors:")
            for e in report.errors:
                print(f"  [X] {e}")
        print("============================================================\n")

    elif args.command == "extract":
        data_path = Path(args.data).resolve()
        if not data_path.exists():
            print(f"Error: Clinical dataset file not found: {data_path}")
            sys.exit(1)

        df = pd.read_excel(data_path) if data_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(data_path)

        extractor = ClinicalFeatureExtractor()
        extractor.fit_and_initialize(df)
        res = extractor.extract_representations(df)

        cr = res["clinical_representations"]
        print("\n============================================================")
        print("PHASE 6 CLINICAL REPRESENTATION EXTRACTION")
        print("============================================================")
        print(f"Extracted Records      : {len(res['patient_ids'])}")
        print(f"Representation Matrix  : Shape {tuple(cr.shape)} (Dim {cr.shape[1]})")
        print(f"Patient Identifiers    : {res['patient_ids'][:5]}...")

        out_path = args.output or (get_clinical_outputs_dir() / "clinical_representations.pt")
        torch.save({
            "clinical_representations": cr,
            "patient_ids": res["patient_ids"],
            "schema": extractor.schema.to_dict(),
        }, str(out_path))
        print(f"\nSaved Clinical Representations to: {out_path}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
