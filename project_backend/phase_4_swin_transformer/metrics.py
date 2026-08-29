"""
Medical Evaluation Metrics and Clinical Curve Visualizers for Phase 4.

Calculates comprehensive classification metrics (Accuracy, Balanced Accuracy,
Sensitivity/Recall, Specificity, Precision, F1, ROC-AUC, PR-AUC, Confusion Matrix)
and generates diagnostic plots.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers/Windows headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive clinical evaluation metric bundle.
    """
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: List[List[int]]
    class_names: List[str]
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> EvaluationMetrics:
    """
    Compute full suite of classification and diagnostic metrics.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        y_prob: Predicted probability array [N, num_classes].
        class_names: List of class string names.

    Returns:
        EvaluationMetrics dataclass.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    num_classes = len(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    if class_names is None:
        class_names = [f"class_{i}" for i in range(max(2, num_classes))]

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    # Per-class scores
    p_class = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    r_class = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    f_class = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)

    per_p: Dict[str, float] = {}
    per_r: Dict[str, float] = {}
    per_f: Dict[str, float] = {}
    for i, c_name in enumerate(class_names[:len(p_class)]):
        per_p[c_name] = float(p_class[i])
        per_r[c_name] = float(r_class[i])
        per_f[c_name] = float(f_class[i])

    roc_auc_val: Optional[float] = None
    pr_auc_val: Optional[float] = None

    if y_prob is not None:
        try:
            if len(class_names) == 2 or (y_prob.ndim == 2 and y_prob.shape[1] == 2):
                pos_probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc_auc_val = float(roc_auc_score(y_true_arr, pos_probs))
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, pos_probs)
                pr_auc_val = float(auc(recall_curve, precision_curve))
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                # Multiclass One-vs-Rest ROC-AUC
                roc_auc_val = float(roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            pass

    return EvaluationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision_macro=prec_macro,
        precision_weighted=prec_weighted,
        recall_macro=rec_macro,
        recall_weighted=rec_weighted,
        f1_macro=f1_mac,
        f1_weighted=f1_wt,
        confusion_matrix=cm,
        class_names=class_names,
        per_class_precision=per_p,
        per_class_recall=per_r,
        per_class_f1=per_f,
        roc_auc=roc_auc_val,
        pr_auc=pr_auc_val,
    )


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names[:cm_arr.shape[1]],
        yticklabels=class_names[:cm_arr.shape[0]],
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Union[str, Path],
    title: str = "ROC Curve",
) -> None:
    """Plot and save ROC curve for binary classification."""
    pos_probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    roc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Union[str, Path],
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot and save PR curve for binary classification."""
    pos_probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    precision, recall, _ = precision_recall_curve(y_true, pos_probs)
    pr_val = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="purple", lw=2, label=f"PR (AUC = {pr_val:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
