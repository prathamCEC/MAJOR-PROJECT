"""
Tests for MultiTaskMetricsCalculator.
"""

import numpy as np
import pytest
import torch

from phase_8_multitask_prediction.metrics import MultiTaskMetricsCalculator


def test_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = MultiTaskMetricsCalculator.calculate_single_task_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        task_name="test_task",
    )

    assert metrics["test_task_accuracy"] == 1.0
    assert metrics["test_task_precision"] == 1.0
    assert metrics["test_task_recall"] == 1.0
    assert metrics["test_task_f1"] == 1.0
    assert metrics["test_task_specificity"] == 1.0
    assert metrics["test_task_sample_count"] == 4


def test_metrics_handles_masked_sentinels():
    # Negative values are filtered as missing
    y_true = np.array([-1, 0, 1, -1, 0])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.4, 0.2, 0.8, 0.9, 0.6])

    metrics = MultiTaskMetricsCalculator.calculate_single_task_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        task_name="stroke",
    )

    # 3 valid samples: true=[0, 1, 0], pred=[0, 1, 1]
    assert metrics["stroke_sample_count"] == 3
    assert np.isclose(metrics["stroke_accuracy"], 2 / 3)


def test_multitask_metrics_dictionary():
    st_true = torch.tensor([0, 1, 1, 0])
    st_pred = torch.tensor([0, 1, 0, 0])
    st_prob = torch.tensor([0.2, 0.7, 0.4, 0.1])

    al_true = torch.tensor([1, 0, 1, 0])
    al_pred = torch.tensor([1, 0, 1, 0])
    al_prob = torch.tensor([0.9, 0.1, 0.8, 0.3])

    m = MultiTaskMetricsCalculator.calculate_multitask_metrics(
        stroke_true=st_true,
        stroke_pred=st_pred,
        stroke_prob=st_prob,
        alzheimer_true=al_true,
        alzheimer_pred=al_pred,
        alzheimer_prob=al_prob,
    )

    assert "stroke_accuracy" in m
    assert "alzheimer_accuracy" in m
    assert m["alzheimer_accuracy"] == 1.0
