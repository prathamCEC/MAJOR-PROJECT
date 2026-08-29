"""
Command Line Interface for Phase 9 Monte Carlo Dropout & Uncertainty Estimation.

Provides CLI utilities to inspect sampling configurations and perform stochastic
uncertainty estimation on saved Unified Patient Representation (UPR) tensors.
"""

import argparse
from pathlib import Path
import sys
import torch

from .config import UncertaintyConfig, get_default_uncertainty_config, get_phase9_outputs_dir
from .engine import MCDropoutUncertaintyEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9 — Monte Carlo Dropout & Model Confidence Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print Phase 9 Monte Carlo Dropout configuration summary")

    # 2. Estimate Command
    est_parser = subparsers.add_parser("estimate", help="Estimate Stroke & Alzheimer's uncertainty using MC Dropout")
    est_parser.add_argument("--upr", type=str, required=True, help="Path to Unified Patient Representation tensor (.pt)")
    est_parser.add_argument("--samples", type=int, default=30, help="Number of MC stochastic forward passes (default: 30)")
    est_parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold (default: 0.5)")
    est_parser.add_argument("--output", type=str, default=None, help="Path to save uncertainty results (.pt)")
    est_parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to Phase 8 model checkpoint (.pth)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_uncertainty_config()
        print("\n============================================================")
        print("PHASE 9 — MONTE CARLO DROPOUT & UNCERTAINTY ESTIMATION")
        print("============================================================")
        print(f"Default MC Samples       : {cfg.mc_samples}")
        print(f"Classification Threshold : {cfg.classification_threshold}")
        print(f"Uncertainty Scale Factor : {cfg.uncertainty_scale}")
        print(f"Numerical Epsilon        : {cfg.epsilon}")
        print(f"Device                   : {cfg.get_device()}")
        print("------------------------------------------------------------")
        print("Confidence Formula       : 1.0 - clamp(variance / scale, 0, 1)")
        print("Entropy Formula          : -p*ln(p) - (1-p)*ln(1-p)")
        print("============================================================")
        print("DISCLAIMER: RESEARCH UNCERTAINTY ESTIMATE ONLY — NOT CLINICAL DIAGNOSIS")
        print("============================================================\n")

    elif args.command == "estimate":
        upr_path = Path(args.upr).resolve()
        if not upr_path.exists():
            print(f"Error: UPR file not found at: {upr_path}")
            sys.exit(1)

        data = torch.load(str(upr_path), map_location="cpu", weights_only=False)
        upr_tensor = data["upr"] if isinstance(data, dict) and "upr" in data else data
        if upr_tensor.ndim == 1:
            upr_tensor = upr_tensor.unsqueeze(0)

        cfg = get_default_uncertainty_config()
        cfg.mc_samples = args.samples
        cfg.classification_threshold = args.threshold

        engine = MCDropoutUncertaintyEngine(
            config=cfg,
            checkpoint_path=args.checkpoint,
        )

        res = engine.estimate_uncertainty(
            upr=upr_tensor,
            mc_samples=args.samples,
            threshold=args.threshold,
            store_mc_predictions=True,
        )

        st = res["stroke"]
        al = res["alzheimer"]
        n_samples = upr_tensor.shape[0]

        print("\n============================================================")
        print("PHASE 9 MONTE CARLO DROPOUT UNCERTAINTY RESULTS")
        print("============================================================")
        print(f"Evaluated Samples : {n_samples}")
        print(f"MC Forward Passes : {args.samples}")
        print(f"Decision Thresh   : {args.threshold}")
        if not res["is_trained_checkpoint"]:
            print("Model Status      : UNTRAINED ARCHITECTURE (Software validation only)")
        print("------------------------------------------------------------")

        for i in range(min(5, n_samples)):
            print(f"Sample {i+1}:")
            print("  STROKE:")
            print(f"    Mean Probability  : {st['mc_mean_probability'][i].item():.4f}")
            print(f"    Variance (sigma^2): {st['mc_variance'][i].item():.6f}")
            print(f"    Std Dev (sigma)   : {st['mc_std'][i].item():.4f}")
            print(f"    Predictive Entropy: {st['predictive_entropy'][i].item():.4f}")
            print(f"    Confidence Score  : {st['confidence_percent'][i].item():.2f}%")
            print(f"    Predicted Class   : {st['prediction'][i].item()}")
            print("  ALZHEIMER'S:")
            print(f"    Mean Probability  : {al['mc_mean_probability'][i].item():.4f}")
            print(f"    Variance (sigma^2): {al['mc_variance'][i].item():.6f}")
            print(f"    Std Dev (sigma)   : {al['mc_std'][i].item():.4f}")
            print(f"    Predictive Entropy: {al['predictive_entropy'][i].item():.4f}")
            print(f"    Confidence Score  : {al['confidence_percent'][i].item():.2f}%")
            print(f"    Predicted Class   : {al['prediction'][i].item()}")
            print("------------------------------------------------------------")

        if n_samples > 5:
            print(f"  ... ({n_samples - 5} additional samples evaluated)")

        print("============================================================")
        print("DISCLAIMER: Model uncertainty is a research metric only.")
        print("============================================================\n")

        out_path = Path(args.output).resolve() if args.output else (get_phase9_outputs_dir() / "uncertainty_estimates.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, str(out_path))
        print(f"Saved uncertainty estimates to: {out_path}\n")


if __name__ == "__main__":
    main()
