"""
Training Pipeline for Phase 4 Swin Transformer.

Orchestrates modality-specific model training, transfer learning, early stopping,
mixed-precision acceleration, and comprehensive validation tracking.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .enums import DiseaseTask, Modality
from .config import ModalityTrainingConfig, get_modality_config, get_splits_dir
from .dataset import RetinalDataset, create_dataloader
from .models.swin_factory import create_swin_model
from .checkpoint import CheckpointManager
from .metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from .split_dataset import create_dataset_splits, load_dataset_splits
from .utils import compute_class_weights, create_experiment_dir, get_device, set_seed
from .validation import DatasetValidator


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float, Dict]:
    """Evaluate model on validation or test dataset."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.append(probs)
            total += labels.size(0)

    val_loss = total_loss / max(1, total)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = calculate_metrics(all_labels, all_preds, probs_arr, class_names=class_names)
    return val_loss, metrics.accuracy, metrics.to_dict()


def train_swin(
    modality: Union[str, Modality],
    data_source: Union[str, Path, pd.DataFrame],
    task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS,
    config: Optional[ModalityTrainingConfig] = None,
    experiment_dir: Optional[Union[str, Path]] = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Complete training pipeline execution.
    """
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    task_enum = DiseaseTask.from_str(task) if isinstance(task, str) else task
    cfg = config or get_modality_config(mod_enum)

    set_seed(cfg.random_seed)
    device = get_device(cfg.device)

    # 1. Setup Experiment Output Directory
    exp_dir = Path(experiment_dir).resolve() if experiment_dir else create_experiment_dir(mod_enum)
    ckpt_mgr = CheckpointManager(exp_dir)

    logging.basicConfig(
        filename=exp_dir / "train.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(f"Initiating Training for Modality={mod_enum.value.upper()}, Task={task_enum.value.upper()}")

    # 2. Dataset Preparation & Partitioning
    if isinstance(data_source, pd.DataFrame):
        full_df = data_source
    else:
        path = Path(data_source).resolve()
        if path.is_file() and path.suffix.lower() == ".csv":
            full_df = pd.read_csv(path)
        else:
            ds_temp = RetinalDataset.from_folder(path, modality=mod_enum)
            full_df = pd.DataFrame([
                {
                    "image_path": str(it.image_path),
                    "modality": it.modality.value,
                    "label": it.label,
                    "class_name": it.class_name,
                    "patient_id": it.patient_id,
                }
                for it in ds_temp.items
            ])

    # Check verified labels
    if "label" not in full_df.columns or full_df["label"].nunique() < 2:
        msg = (
            f"Cannot train supervised model for {mod_enum.value.upper()}: "
            "Dataset does not contain verified class labels (minimum 2 classes required)."
        )
        logging.error(msg)
        raise ValueError(msg)

    # Check/create splits
    splits = load_dataset_splits(mod_enum, task_enum)
    if splits is None:
        train_df, val_df, test_df = create_dataset_splits(
            full_df, modality=mod_enum, task=task_enum, random_seed=cfg.random_seed
        )
    else:
        train_df, val_df, test_df = splits

    # Build Class Mapping
    unique_classes = sorted(full_df["class_name"].unique()) if "class_name" in full_df.columns else [f"class_{i}" for i in sorted(full_df["label"].unique())]
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    cfg.num_classes = num_classes

    ckpt_mgr.save_config(cfg, class_to_idx)

    # Build Datasets & Loaders
    train_dataset = RetinalDataset.from_csv(
        get_splits_dir() / f"{mod_enum.value}_{task_enum.value}_train.csv",
        modality=mod_enum,
        is_training=True,
        image_size=cfg.image_size,
    )
    val_dataset = RetinalDataset.from_csv(
        get_splits_dir() / f"{mod_enum.value}_{task_enum.value}_val.csv",
        modality=mod_enum,
        is_training=False,
        image_size=cfg.image_size,
    )
    test_dataset = RetinalDataset.from_csv(
        get_splits_dir() / f"{mod_enum.value}_{task_enum.value}_test.csv",
        modality=mod_enum,
        is_training=False,
        image_size=cfg.image_size,
    )

    train_loader = create_dataloader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = create_dataloader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = create_dataloader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 3. Model, Loss, Optimizer, Scheduler Setup
    model = create_swin_model(
        modality=mod_enum,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        model_name=cfg.model_name,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    # Compute training class weights
    class_weights = compute_class_weights(train_df["label"].values, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if (cfg.mixed_precision and device.type == "cuda") else None

    start_epoch = 1
    best_val_f1 = -1.0
    patience_counter = 0

    if resume_checkpoint:
        ckpt_data = CheckpointManager.load_checkpoint(resume_checkpoint, model, optimizer, scheduler, device=device)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_val_f1 = ckpt_data.get("best_metric", -1.0)
        logging.info(f"Resumed from epoch {start_epoch} with best F1={best_val_f1:.4f}")

    # 4. Training Loop
    history_file = exp_dir / "training_history.csv"
    history_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "val_precision", "val_recall", "lr"]

    with open(history_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(history_fields)

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Unfreeze backbone if scheduled
        if cfg.freeze_backbone and epoch == cfg.unfreeze_at_epoch:
            logging.info(f"Epoch {epoch}: Unfreezing Swin backbone for end-to-end fine-tuning.")
            model.unfreeze_backbone()
            # Recreate optimizer with lower learning rate for fine-tuning
            optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate * 0.1, weight_decay=cfg.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - epoch + 1, eta_min=1e-7)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_metrics = evaluate_epoch(model, val_loader, criterion, device, class_names=unique_classes)
        scheduler.step()

        val_f1 = val_metrics["f1_macro"]
        val_p = val_metrics["precision_macro"]
        val_r = val_metrics["recall_macro"]
        current_lr = optimizer.param_groups[0]["lr"]

        # Append to CSV history
        with open(history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_p, val_r, current_lr])

        logging.info(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_val_f1,
            class_mapping=class_to_idx,
            modality=mod_enum,
            task=task_enum,
            is_best=is_best,
        )

        if patience_counter >= cfg.early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs (patience={cfg.early_stopping_patience}).")
            break

    # 5. Final Evaluation on Test Set using Best Checkpoint
    logging.info("Executing final evaluation on Test Set using best model checkpoint...")
    CheckpointManager.load_checkpoint(ckpt_mgr.best_model_path, model, device=device)
    test_loss, test_acc, test_metrics = evaluate_epoch(model, test_loader, criterion, device, class_names=unique_classes)

    # Save metrics JSON
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Generate diagnostic plots
    plot_confusion_matrix(test_metrics["confusion_matrix"], unique_classes, exp_dir / "confusion_matrix.png")
    
    return {
        "experiment_dir": str(exp_dir),
        "best_model_path": str(ckpt_mgr.best_model_path),
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — Swin Transformer Model Training")
    parser.add_argument("--modality", type=str, required=True, choices=["octa", "octb", "fundus"], help="Target modality")
    parser.add_argument("--task", type=str, default="alzheimers", choices=["stroke", "alzheimers", "multi_disease"], help="Disease task")
    parser.add_argument("--data", type=str, default=None, help="Dataset CSV or class folder path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
