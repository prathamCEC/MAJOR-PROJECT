"""
Model Evaluation Module for Phase 4 Swin Transformer.

Evaluates trained Swin Transformer checkpoints on independent test sets,
computes clinical diagnostic metrics, and generates publication-grade evaluation curves.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .enums import DiseaseTask, Modality
from .config import get_modality_config, get_splits_dir
from .dataset import RetinalDataset, create_dataloader
from .models.swin_factory import create_swin_model
from .checkpoint import CheckpointManager
from .metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from .utils import get_device


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    test_data_path: Optional[Union[str, Path]] = None,
    modality: Optional[Union[str, Modality]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Evaluate a saved model checkpoint on the test set.
    """
    ckpt_p = Path(checkpoint_path).resolve()
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    out_p = Path(output_dir).resolve() if output_dir else ckpt_p.parent
    out_p.mkdir(parents=True, exist_ok=True)

    device = get_device("auto")

    # 1. Load Checkpoint Metadata
    raw_ckpt = torch.load(str(ckpt_p), map_location="cpu")
    mod_str = modality or raw_ckpt.get("modality", "octa")
    mod_enum = Modality.from_str(mod_str)
    task_str = raw_ckpt.get("task", "alzheimers")
    task_enum = DiseaseTask.from_str(task_str)
    class_mapping: Dict[str, int] = raw_ckpt.get("class_mapping", {"normal": 0, "disease": 1})
    num_classes = len(class_mapping)

    # Invert class mapping: idx -> class_name
    idx_to_class = {v: k for k, v in class_mapping.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # 2. Build Model & Restore Weights
    cfg = get_modality_config(mod_enum)
    model = create_swin_model(
        modality=mod_enum,
        num_classes=num_classes,
        pretrained=False,
        model_name=cfg.model_name,
    )
    CheckpointManager.load_checkpoint(ckpt_p, model, device=device)
    model.to(device)
    model.eval()

    # 3. Locate Test Data
    if test_data_path:
        test_p = Path(test_data_path).resolve()
    else:
        test_p = get_splits_dir() / f"{mod_enum.value}_{task_enum.value}_test.csv"

    if not test_p.exists():
        raise FileNotFoundError(f"Test split dataset manifest not found at: {test_p}")

    test_dataset = RetinalDataset.from_csv(
        test_p,
        modality=mod_enum,
        is_training=False,
        image_size=cfg.image_size,
    )
    test_loader = create_dataloader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 4. Run Evaluation
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else None
    eval_metrics = calculate_metrics(all_labels, all_preds, probs_arr, class_names=class_names)
    metrics_dict = eval_metrics.to_dict()

    # 5. Save Artifacts & Curves
    with open(out_p / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    plot_confusion_matrix(
        eval_metrics.confusion_matrix,
        class_names,
        out_p / "confusion_matrix.png",
        title=f"{mod_enum.value.upper()} Confusion Matrix ({task_enum.value.upper()})",
    )

    if probs_arr is not None and num_classes == 2:
        plot_roc_curve(
            np.array(all_labels),
            probs_arr,
            out_p / "roc_curve.png",
            title=f"{mod_enum.value.upper()} ROC Curve",
        )
        plot_precision_recall_curve(
            np.array(all_labels),
            probs_arr,
            out_p / "precision_recall_curve.png",
            title=f"{mod_enum.value.upper()} PR Curve",
        )

    # 6. Format Report
    print("\n============================================================")
    print(f"PHASE 4 EVALUATION REPORT ({mod_enum.value.upper()})")
    print("============================================================")
    print(f"Total Test Samples     : {len(all_labels)}")
    print(f"Accuracy               : {eval_metrics.accuracy:.4f}")
    print(f"Balanced Accuracy      : {eval_metrics.balanced_accuracy:.4f}")
    print(f"Macro Precision        : {eval_metrics.precision_macro:.4f}")
    print(f"Macro Recall (Sens.)   : {eval_metrics.recall_macro:.4f}")
    print(f"Macro F1-Score         : {eval_metrics.f1_macro:.4f}")
    if eval_metrics.roc_auc is not None:
        print(f"ROC-AUC                : {eval_metrics.roc_auc:.4f}")
    if eval_metrics.pr_auc is not None:
        print(f"PR-AUC                 : {eval_metrics.pr_auc:.4f}")
    print("\nPer-Class Metrics:")
    for c_name in class_names:
        p = eval_metrics.per_class_precision.get(c_name, 0.0)
        r = eval_metrics.per_class_recall.get(c_name, 0.0)
        f = eval_metrics.per_class_f1.get(c_name, 0.0)
        print(f"  - {c_name:<18} | Prec: {p:.4f} | Rec: {r:.4f} | F1: {f:.4f}")
    print("============================================================\n")

    return metrics_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — Swin Transformer Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument("--modality", type=str, default=None, choices=["octa", "octb", "fundus"], help="Modality")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test CSV manifest")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation plots and metrics")

    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        modality=args.modality,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
