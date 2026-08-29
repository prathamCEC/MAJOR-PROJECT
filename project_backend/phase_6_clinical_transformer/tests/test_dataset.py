"""
Tests for ClinicalTabularDataset and patient-isolated splitting.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from phase_6_clinical_transformer.dataset import (
    ClinicalTabularDataset,
    create_clinical_dataloader,
    patient_level_split,
)


def test_clinical_tabular_dataset():
    num_mat = np.random.randn(10, 2).astype(np.float32)
    cat_mat = np.random.randint(0, 3, (10, 4)).astype(np.int64)
    pids = [f"P{i:02d}" for i in range(10)]

    dataset = ClinicalTabularDataset(num_mat, cat_mat, patient_ids=pids)
    assert len(dataset) == 10

    item = dataset[0]
    assert item["numerical_features"].shape == (2,)
    assert item["categorical_features"].shape == (4,)
    assert item["patient_id"] == "P00"

    loader = create_clinical_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["numerical_features"].shape == (4, 2)
    assert batch["categorical_features"].shape == (4, 4)
    assert len(batch["patient_id"]) == 4


def test_patient_level_split_strict_isolation():
    # 30 records across 10 patients (3 records per patient)
    records = []
    for p_idx in range(10):
        pid = f"PATIENT_{p_idx:02d}"
        for r in range(3):
            records.append({
                "ID#": pid,
                "BMI": 22.0 + p_idx,
                "Education": 12,
                "Gender": p_idx % 2,
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = patient_level_split(
        df,
        patient_id_col="ID#",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    train_pids = set(train_df["ID#"].unique())
    val_pids = set(val_df["ID#"].unique())
    test_pids = set(test_df["ID#"].unique())

    # Verify zero overlap between patient sets
    assert len(train_pids.intersection(val_pids)) == 0
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(val_pids.intersection(test_pids)) == 0
    assert len(train_pids) + len(val_pids) + len(test_pids) == 10
