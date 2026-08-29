"""
Command Line Interface for Phase 7 Retina-Clinical Cross-Attention Fusion.

CLI supporting architecture summary and multimodal Unified Patient Representation (UPR) extraction.
"""

import argparse
from pathlib import Path
import sys
import torch

from .config import RetinaClinicalConfig, get_default_retina_clinical_config, get_phase7_outputs_dir
from .fusion_model import RetinaClinicalFusionModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 — Retina-Clinical Cross-Attention Fusion & Unified Patient Representation (UPR)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print Phase 7 fusion architecture summary")

    # 2. Fuse Command
    fuse_parser = subparsers.add_parser("fuse", help="Fuse Retinal (URR) and Clinical (CR) representations into UPR")
    fuse_parser.add_argument("--retinal", type=str, required=True, help="Path to retinal representation tensor (.pt)")
    fuse_parser.add_argument("--clinical", type=str, required=True, help="Path to clinical representation tensor (.pt)")
    fuse_parser.add_argument("--output", type=str, default=None, help="Output path for saved UPR tensor (.pt)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_retina_clinical_config()
        print("\n============================================================")
        print("PHASE 7 — RETINA–CLINICAL CROSS-ATTENTION & UPR ARCHITECTURE")
        print("============================================================")
        print(f"Retinal Input Dimension : {cfg.retinal_input_dim} (Phase 5 URR)")
        print(f"Clinical Input Dimension: {cfg.clinical_input_dim} (Phase 6 CR)")
        print(f"Common Embedding Dim    : {cfg.common_embed_dim}")
        print(f"Cross-Attention Heads   : {cfg.num_heads}")
        print(f"Bidirectional Layers    : {cfg.num_layers} blocks")
        print(f"Feed-Forward Dim        : {cfg.ffn_dim}")
        print(f"Pooling Strategy        : {cfg.pooling_strategy.upper()}")
        print(f"Fusion Strategy         : {cfg.fusion_strategy.upper()} Gating")
        print(f"UPR Output Dimension    : {cfg.upr_dim} (For Phase 8)")
        print(f"Device                  : {cfg.get_device()}")
        print("============================================================\n")

    elif args.command == "fuse":
        ret_path = Path(args.retinal).resolve()
        clin_path = Path(args.clinical).resolve()

        if not ret_path.exists():
            print(f"Error: Retinal representation file not found at: {ret_path}")
            sys.exit(1)
        if not clin_path.exists():
            print(f"Error: Clinical representation file not found at: {clin_path}")
            sys.exit(1)

        # Load tensors
        ret_data = torch.load(str(ret_path), map_location="cpu", weights_only=False)
        ret_tensor = ret_data["urr"] if isinstance(ret_data, dict) and "urr" in ret_data else (
            ret_data["retinal_representation"] if isinstance(ret_data, dict) and "retinal_representation" in ret_data else ret_data
        )

        clin_data = torch.load(str(clin_path), map_location="cpu", weights_only=False)
        clin_tensor = clin_data["clinical_representations"] if isinstance(clin_data, dict) and "clinical_representations" in clin_data else (
            clin_data["clinical_representation"] if isinstance(clin_data, dict) and "clinical_representation" in clin_data else clin_data
        )

        # Batch alignment check / slicing for demonstration if needed
        if ret_tensor.ndim == 1:
            ret_tensor = ret_tensor.unsqueeze(0)
        if clin_tensor.ndim == 1:
            clin_tensor = clin_tensor.unsqueeze(0)

        # Align batch size for demonstration if single retinal vs full clinical cohort
        min_b = min(ret_tensor.shape[0], clin_tensor.shape[0])
        ret_tensor = ret_tensor[:min_b]
        clin_tensor = clin_tensor[:min_b]

        model = RetinaClinicalFusionModel()
        model.eval()

        with torch.no_grad():
            res = model(retinal_representation=ret_tensor, clinical_representation=clin_tensor)

        upr = res["upr"]
        print("\n============================================================")
        print("PHASE 7 UNIFIED PATIENT REPRESENTATION (UPR) GENERATED")
        print("============================================================")
        print(f"Input Retinal Shape   : {tuple(ret_tensor.shape)}")
        print(f"Input Clinical Shape  : {tuple(clin_tensor.shape)}")
        print(f"Output UPR Shape      : {tuple(upr.shape)} (Dim {upr.shape[1]})")
        print(f"Multimodal Gate Mean  : {res['gate_weights'].mean().item():.4f}")

        out_path = Path(args.output).resolve() if args.output else (get_phase7_outputs_dir() / "unified_patient_representation.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "upr": upr,
            "gate_weights": res["gate_weights"],
            "config": model.config.to_dict(),
        }, str(out_path))
        print(f"\nSaved UPR tensor to: {out_path}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
