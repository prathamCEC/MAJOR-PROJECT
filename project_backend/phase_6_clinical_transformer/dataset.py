"""
Dataset and DataLoader Module for Clinical Tabular Data.

Provides PyTorch Dataset implementations for structured clinical data and
patient-isolated splitting to prevent clinical data leakage.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class ClinicalTabularDataset(Dataset):
    """
    PyTorch Dataset for Preprocessed Patient Clinical Data.
    """

    def __init__(
        self,
        numerical_matrix: np.ndarray,
        categorical_matrix: np.ndarray,
        patient_ids: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
    ):
        self.num_samples = numerical_matrix.shape[0] if numerical_matrix.shape[1] > 0 else categorical_matrix.shape[0]
        self.x_num = torch.tensor(numerical_matrix, dtype=torch.float32)
        self.x_cat = torch.tensor(categorical_matrix, dtype=torch.long)
        self.patient_ids = patient_ids or [f"patient_{i}" for i in range(self.num_samples)]
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "numerical_features": self.x_num[idx],
            "categorical_features": self.x_cat[idx],
            "patient_id": self.patient_ids[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def create_clinical_dataloader(
    dataset: ClinicalTabularDataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a robust DataLoader for ClinicalTabularDataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def patient_level_split(
    df: pd.DataFrame,
    patient_id_col: Optional[str] = "ID#",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split clinical DataFrame into train, val, and test sets with strict patient isolation.

    Guarantees that all records from any given patient belong exclusively to one split.
    """
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    if patient_id_col and patient_id_col in df.columns:
        # GroupShuffleSplit based on patient identifiers
        patients = df[patient_id_col].values
        unique_patients = np.unique(patients)

        if len(unique_patients) >= 3:
            # 1. Split Train vs (Val + Test)
            gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
            train_idx, temp_idx = next(gss1.split(df, groups=patients))

            train_df = df.iloc[train_idx].copy().reset_index(drop=True)
            temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

            # 2. Split Val vs Test
            val_relative_ratio = val_ratio / (val_ratio + test_ratio)
            gss2 = GroupShuffleSplit(n_splits=1, train_size=val_relative_ratio, random_state=random_seed)
            val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[patient_id_col].values))

            val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
            test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

            return train_df, val_df, test_df

    # Fallback to simple random split if patient IDs not present or too few unique patients
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_seed,
    )
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
