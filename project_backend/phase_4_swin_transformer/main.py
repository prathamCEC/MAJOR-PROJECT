"""
Command Line Interface for Phase 4 Swin Transformer.

Unified entry point supporting dataset audit, splitting, training, evaluation,
inference, and explainability workflows across OCT-A, OCT-B, and Fundus modalities.
"""

import argparse
from pathlib import Path
import sys

from .enums import DiseaseTask, Modality
from .config import get_modality_config, get_splits_dir, get_approved_dataset_dir
from .validation import validate_modality_dataset
from .split_dataset import create_dataset_splits, load_dataset_splits
from .leakage_check import check_splits_leakage
from .train import train_swin
from .evaluate import evaluate_checkpoint
from .inference import SwinInferenceEngine
from .models.swin_factory import create_swin_model
from .explainability import SwinExplainabilityEngine
from .checkpoint import CheckpointManager
from .utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Swin Transformer for Retinal Disease Analysis (OCT-A, OCT-B, Fundus)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Validate Command
    val_parser = subparsers.add_parser("validate", help="Validate dataset labels and images")
    val_parser.add_argument("--modality", type=str, required=True, choices=["octa", "octb", "fundus"], help="Modality")
    val_parser.add_argument("--task", type=str, default="alzheimers", choices=["stroke", "alzheimers", "multi_disease"])
    val_parser.add_argument("--data", type=str, default=None, help="Path to dataset folder or CSV manifest")

    # 2. Split Command
    split_parser = subparsers.add_parser("split", help="Create train/val/test splits")
    split_parser.add_argument("--modality", type=str, required=True, choices=["octa", "octb", "fundus"], help="Modality")
    split_parser.add_argument("--task", type=str, default="alzheimers", choices=["stroke", "alzheimers", "multi_disease"])
    split_parser.add_argument("--data", type=str, required=True, help="Path to full dataset CSV or folder")

    # 3. Train Command
    train_parser = subparsers.add_parser("train", help="Train Swin Transformer model")
    train_parser.add_argument("--modality", type=str, required=True, choices=["octa", "octb", "fundus"], help="Modality")
    train_parser.add_argument("--task", type=str, default="alzheimers", choices=["stroke", "alzheimers", "multi_disease"])
    train_parser.add_argument("--data", type=str, default=None, help="Dataset CSV or class directory")
    train_parser.add_argument("--epochs", type=int, default=None, help="Epochs")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # 4. Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate checkpoint on test set")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    eval_parser.add_argument("--modality", type=str, default=None, choices=["octa", "octb", "fundus"])
    eval_parser.add_argument("--test-data", type=str, default=None, help="Path to test CSV manifest")
    eval_parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")

    # 5. Inference Command
    inf_parser = subparsers.add_parser("inference", help="Run model inference on single image or batch")
    inf_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    inf_parser.add_argument("--modality", type=str, default=None, choices=["octa", "octb", "fundus"])
    inf_parser.add_argument("--image", type=str, default=None, help="Path to single image")
    inf_parser.add_argument("--input", type=str, default=None, help="Directory for batch prediction")
    inf_parser.add_argument("--output-csv", type=str, default=None, help="Path to save predictions CSV")

    # 6. Explain Command
    exp_parser = subparsers.add_parser("explain", help="Generate activation heatmaps")
    exp_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    exp_parser.add_argument("--modality", type=str, required=True, choices=["octa", "octb", "fundus"])
    exp_parser.add_argument("--image", type=str, required=True, help="Path to image")
    exp_parser.add_argument("--output", type=str, default=None, help="Path to save explanation image")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "validate":
        data_src = args.data or get_approved_dataset_dir(args.modality)
        stats = validate_modality_dataset(data_src, modality=args.modality, task=args.task)
        print(f"\nValidation Result for {args.modality.upper()} ({args.task.upper()}):")
        print(f"Total Images: {stats.total_images}, Valid: {stats.valid_images}, Missing: {stats.missing_files}, Corrupt: {stats.corrupted_files}")
        print(f"Verified Labels Exist: {stats.has_verified_labels}")
        print(f"Classes: {stats.class_distribution}")
        if stats.error_messages:
            for err in stats.error_messages:
                print(f"  [!] {err}")

    elif args.command == "split":
        data_src = Path(args.data).resolve()
        import pandas as pd
        if data_src.is_file() and data_src.suffix.lower() == ".csv":
            df = pd.read_csv(data_src)
        else:
            from .dataset import RetinalDataset
            ds = RetinalDataset.from_folder(data_src, modality=args.modality)
            df = pd.DataFrame([
                {"image_path": str(it.image_path), "modality": it.modality.value, "label": it.label, "class_name": it.class_name, "patient_id": it.patient_id}
                for it in ds.items
            ])

        train_df, val_df, test_df = create_dataset_splits(df, modality=args.modality, task=args.task)
        leakage = check_splits_leakage(train_df, val_df, test_df)
        print(leakage.format_summary())

    elif args.command == "train":
        cfg = get_modality_config(args.modality)
        if args.epochs:
            cfg.epochs = args.epochs
        if args.batch_size:
            cfg.batch_size = args.batch_size
        if args.lr:
            cfg.learning_rate = args.lr

        data_src = args.data if args.data else (get_splits_dir() / f"{args.modality}_{args.task}_train.csv")
        train_swin(
            modality=args.modality,
            data_source=data_src,
            task=args.task,
            config=cfg,
            resume_checkpoint=args.resume,
        )

    elif args.command == "evaluate":
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            modality=args.modality,
            output_dir=args.output_dir,
        )

    elif args.command == "inference":
        engine = SwinInferenceEngine(checkpoint_path=args.checkpoint, modality=args.modality)
        if args.image:
            res = engine.predict_image(args.image)
            print(f"\nPredicted Class: {res.predicted_class} (Confidence: {res.confidence * 100:.2f}%)")
            for c_name, p in sorted(res.probabilities.items()):
                print(f"  - {c_name:<18}: {p * 100:6.2f}%")
        elif args.input:
            out_csv = args.output_csv or (Path(args.input) / "predictions.csv")
            results = engine.predict_batch(args.input, output_csv_path=out_csv)
            print(f"Batch inference complete for {len(results)} images. Saved: {out_csv}")

    elif args.command == "explain":
        engine = SwinInferenceEngine(checkpoint_path=args.checkpoint, modality=args.modality)
        exp_engine = SwinExplainabilityEngine(engine.model, modality=args.modality)
        out_p = args.output or (Path(args.checkpoint).parent / "explanations" / f"{Path(args.image).stem}_explanation.png")
        saved = exp_engine.save_explanation(args.image, out_p)
        print(f"Saved explainability heatmap to: {saved}")


if __name__ == "__main__":
    main()
