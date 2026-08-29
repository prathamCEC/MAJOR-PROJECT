"""
Tests for clinical metrics calculation and plotting utilities.
"""

from pathlib import Path
import numpy as np
import pytest

from phase_4_swin_transformer.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)


def test_calculate_metrics_binary():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    class_names = ["normal", "disease"]

    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names=class_names)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]


def test_plot_generators(tmp_path: Path):
    cm = [[5, 1], [0, 6]]
    class_names = ["normal", "disease"]
    cm_path = tmp_path / "cm.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]])
    roc_path = tmp_path / "roc.png"
    plot_roc_curve(y_true, y_prob, roc_path)
    assert roc_path.exists()
    assert roc_path.stat().st_size > 0
