"""
Command Line Interface for Phase 5 Retinal Multimodal Fusion.

Unified CLI supporting multimodal retinal feature fusion, validation, and testing.
"""

import argparse
from pathlib import Path
import sys
import torch

from .config import FusionConfig, get_default_fusion_config, get_fusion_outputs_dir
from .fusion_model import RetinalMultimodalFusionModel
from .feature_loader import Phase4FeatureExtractor
from .validation import validate_input_features, validate_urr_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 — Dynamic Modality Reliability Attention (DMRA) & Unified Retinal Representation (URR)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. Fuse Command
    fuse_parser = subparsers.add_parser("fuse", help="Run multimodal fusion on retinal image scans")
    fuse_parser.add_argument("--octa", type=str, default=None, help="Path to OCT-A image")
    fuse_parser.add_argument("--octb", type=str, default=None, help="Path to OCT-B image")
    fuse_parser.add_argument("--fundus", type=str, default=None, help="Path to Fundus image")
    fuse_parser.add_argument("--output", type=str, default=None, help="Output path for URR tensor (.pt)")

    # 2. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print Phase 5 Fusion Model Architecture")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_fusion_config()
        model = RetinalMultimodalFusionModel(config=cfg)
        print("\n============================================================")
        print("PHASE 5 — RETINAL MULTIMODAL FUSION ARCHITECTURE (DMRA + URR)")
        print("============================================================")
        print(f"Modalities Supported   : {cfg.modalities}")
        print(f"Input Feature Dims     : {cfg.input_dims}")
        print(f"Projection Embed Dim   : {cfg.embed_dim}")
        print(f"Cross-Attention Heads  : {cfg.num_heads}")
        print(f"Fusion Transformer Lvls: {cfg.num_fusion_layers}")
        print(f"Feed-Forward Dim       : {cfg.ffn_dim}")
        print(f"URR Output Dim         : {cfg.urr_dim}")
        print(f"Reliability Scorer     : DMRA (Dynamic Modality Reliability Attention)")
        print(f"Device                 : {cfg.get_device()}")
        print("============================================================\n")

    elif args.command == "fuse":
        patient_scans = {}
        if args.octa:
            patient_scans["octa"] = args.octa
        if args.octb:
            patient_scans["octb"] = args.octb
        if args.fundus:
            patient_scans["fundus"] = args.fundus

        if not patient_scans:
            print("Error: Please provide at least one modality image (--octa, --octb, or --fundus).")
            sys.exit(1)

        extractor = Phase4FeatureExtractor(device="cpu", pretrained_backbone=False)
        feats, masks = extractor.extract_multimodal_patient_features(patient_scans, pool=False)

        cfg = get_default_fusion_config()
        cfg.device = "cpu"
        model = RetinalMultimodalFusionModel(config=cfg)
        model.eval()

        with torch.no_grad():
            res = model(modality_features=feats, modality_mask=masks)

        urr = res["urr"]
        weights = res["modality_weights"]

        print("\n============================================================")
        print("PHASE 5 MULTIMODAL FUSION RESULT")
        print("============================================================")
        print(f"Available Modalities   : {list(feats.keys())}")
        print(f"Unified Retinal Vector : Shape {tuple(urr.shape)} (Dim {cfg.urr_dim})")
        print("Learned Modality Weights:")
        for m, w in weights.items():
            print(f"  - {m.upper():<8}: {w.item():.4f} ({w.item() * 100:.2f}%)")

        out_path = args.output or (get_fusion_outputs_dir() / "fused_urr.pt")
        torch.save({
            "urr": urr.cpu(),
            "modality_weights": {m: w.cpu().item() for m, w in weights.items()},
        }, str(out_path))
        print(f"\nSaved Unified Retinal Representation (URR) to: {out_path}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
