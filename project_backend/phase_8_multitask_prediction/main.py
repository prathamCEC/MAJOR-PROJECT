"""
Command Line Interface for Phase 8 Multi-Task Disease Prediction Network.

Provides architecture summaries, disease prediction from UPR tensors, and research evaluations.
"""

import argparse
from pathlib import Path
import sys
import torch

from .config import MultiTaskConfig, get_default_multitask_config, get_phase8_outputs_dir
from .model import MultiTaskDiseasePredictionNetwork
from .inference import EndToEndDiseasePredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 — Multi-Task Disease Prediction Network (Stroke + Alzheimer's Disease)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Summary Command
    sum_parser = subparsers.add_parser("summary", help="Print Phase 8 multi-task network architecture summary")

    # 2. Predict Command
    pred_parser = subparsers.add_parser("predict", help="Generate Stroke and Alzheimer's predictions from UPR tensor")
    pred_parser.add_argument("--upr", type=str, required=True, help="Path to Unified Patient Representation tensor (.pt)")
    pred_parser.add_argument("--output", type=str, default=None, help="Path to save prediction results (.pt)")
    pred_parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold (default: 0.5)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        cfg = get_default_multitask_config()
        print("\n============================================================")
        print("PHASE 8 — MULTI-TASK DISEASE PREDICTION NETWORK")
        print("============================================================")
        print(f"Input UPR Dimension      : {cfg.upr_dim} (from Phase 7)")
        print(f"Shared Trunk Dimension   : {cfg.shared_hidden_dim}")
        print(f"Task Head Hidden Dim     : {cfg.task_hidden_dim}")
        print(f"Dropout Probability      : {cfg.dropout}")
        print(f"Stroke Loss Weight (lambda)   : {cfg.stroke_loss_weight}")
        print(f"Alzheimer's Loss Wt (lambda)  : {cfg.alzheimer_loss_weight}")
        print(f"Classification Threshold : {cfg.classification_threshold}")
        print(f"Device                   : {cfg.get_device()}")
        print("============================================================")
        print("DISCLAIMER: RESEARCH PREDICTION SYSTEM ONLY — NOT CLINICAL DIAGNOSIS")
        print("============================================================\n")

    elif args.command == "predict":
        upr_path = Path(args.upr).resolve()
        if not upr_path.exists():
            print(f"Error: UPR file not found at: {upr_path}")
            sys.exit(1)

        data = torch.load(str(upr_path), map_location="cpu", weights_only=False)
        upr_tensor = data["upr"] if isinstance(data, dict) and "upr" in data else data
        if upr_tensor.ndim == 1:
            upr_tensor = upr_tensor.unsqueeze(0)

        predictor = EndToEndDiseasePredictor()
        res = predictor.predict_from_upr(upr=upr_tensor, threshold=args.threshold)

        st_prob = res["stroke"]["probability"]
        st_pred = res["stroke"]["prediction"]
        al_prob = res["alzheimer"]["probability"]
        al_pred = res["alzheimer"]["prediction"]

        print("\n============================================================")
        print("PHASE 8 MULTI-TASK DISEASE PREDICTIONS")
        print("============================================================")
        print(f"Evaluated Samples        : {upr_tensor.shape[0]}")
        print(f"Decision Threshold       : {args.threshold}")
        print("------------------------------------------------------------")
        for i in range(min(5, upr_tensor.shape[0])):
            print(f"Sample {i+1}:")
            print(f"  Stroke      : Probability {st_prob[i, 0].item():.4f} -> Predicted Class {st_pred[i, 0].item()}")
            print(f"  Alzheimer's : Probability {al_prob[i, 0].item():.4f} -> Predicted Class {al_pred[i, 0].item()}")
        if upr_tensor.shape[0] > 5:
            print(f"  ... ({upr_tensor.shape[0] - 5} additional samples evaluated)")
        print("============================================================")
        print("DISCLAIMER: Model output is research prediction only.")
        print("============================================================\n")

        out_path = Path(args.output).resolve() if args.output else (get_phase8_outputs_dir() / "multitask_predictions.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, str(out_path))
        print(f"Saved prediction results to: {out_path}\n")


if __name__ == "__main__":
    main()
