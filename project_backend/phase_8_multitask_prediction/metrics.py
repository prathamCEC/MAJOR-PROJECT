"""
Clinical Metrics Engine for Multi-Task Disease Evaluation.

Computes accuracy, precision, recall, specificity, F1-score, ROC-AUC, PR-AUC,
and confusion matrix statistics strictly over valid (labelled) patient samples.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import torch


class MultiTaskMetricsCalculator:
    """
    Computes statistical and clinical diagnostic metrics for Stroke and Alzheimer's tasks.
    """

    @staticmethod
    def calculate_single_task_metrics(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        task_name: str = "task",
    ) -> Dict[str, Any]:
        """
        Calculate binary classification metrics for a single task.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy().flatten()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy().flatten()
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.detach().cpu().numpy().flatten()

        # Filter out invalid / missing samples (e.g. negative or NaN)
        valid_idx = (~np.isnan(y_true)) & (y_true >= 0)
        y_true = y_true[valid_idx].astype(int)
        y_pred = y_pred[valid_idx].astype(int)
        y_prob = y_prob[valid_idx].astype(float)

        n_samples = len(y_true)
        if n_samples == 0:
            return {
                f"{task_name}_sample_count": 0,
                f"{task_name}_accuracy": 0.0,
                f"{task_name}_precision": 0.0,
                f"{task_name}_recall": 0.0,
                f"{task_name}_specificity": 0.0,
                f"{task_name}_f1": 0.0,
                f"{task_name}_roc_auc": 0.0,
                f"{task_name}_pr_auc": 0.0,
                f"{task_name}_confusion_matrix": [[0, 0], [0, 0]],
            }

        # Basic Binary Classification Metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion Matrix & Specificity
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC-AUC and PR-AUC (require at least 2 distinct classes)
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                roc_auc = 0.0
            try:
                pr_auc = float(average_precision_score(y_true, y_prob))
            except ValueError:
                pr_auc = 0.0
        else:
            roc_auc = 0.0
            pr_auc = 0.0

        return {
            f"{task_name}_sample_count": n_samples,
            f"{task_name}_accuracy": acc,
            f"{task_name}_precision": prec,
            f"{task_name}_recall": rec,
            f"{task_name}_specificity": spec,
            f"{task_name}_f1": f1,
            f"{task_name}_roc_auc": roc_auc,
            f"{task_name}_pr_auc": pr_auc,
            f"{task_name}_confusion_matrix": cm.tolist(),
        }

    @classmethod
    def calculate_multitask_metrics(
        cls,
        stroke_true: Optional[Union[np.ndarray, torch.Tensor]],
        stroke_pred: Optional[Union[np.ndarray, torch.Tensor]],
        stroke_prob: Optional[Union[np.ndarray, torch.Tensor]],
        alzheimer_true: Optional[Union[np.ndarray, torch.Tensor]],
        alzheimer_pred: Optional[Union[np.ndarray, torch.Tensor]],
        alzheimer_prob: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, Any]:
        """
        Calculate metrics for both Stroke and Alzheimer's disease prediction tasks.
        """
        metrics = {}

        if stroke_true is not None and stroke_pred is not None and stroke_prob is not None:
            metrics.update(
                cls.calculate_single_task_metrics(
                    y_true=stroke_true,
                    y_pred=stroke_pred,
                    y_prob=stroke_prob,
                    task_name="stroke",
                )
            )

        if alzheimer_true is not None and alzheimer_pred is not None and alzheimer_prob is not None:
            metrics.update(
                cls.calculate_single_task_metrics(
                    y_true=alzheimer_true,
                    y_pred=alzheimer_pred,
                    y_prob=alzheimer_prob,
                    task_name="alzheimer",
                )
            )

        return metrics
