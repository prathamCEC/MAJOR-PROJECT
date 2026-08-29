"""
Dataset Splitting Module for Phase 4 Swin Transformer.

Performs stratified, patient-isolated partitioning (70% train, 15% val, 15% test)
and exports reproducible split manifest files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split

from .enums import DiseaseTask, Modality
from .config import get_splits_dir


def create_dataset_splits(
    df: pd.DataFrame,
    modality: Union[str, Modality],
    task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Guarantees patient-level isolation if 'patient_id' column is present.
    Uses stratification on class labels where possible.
    """
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    task_enum = DiseaseTask.from_str(task) if isinstance(task, str) else task
    out_dir = Path(output_dir).resolve() if output_dir else get_splits_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe.")

    has_patients = "patient_id" in df.columns and df["patient_id"].notna().any()
    label_col = "label"

    if has_patients and df["patient_id"].nunique() >= 3:
        # Group-based split by patient_id
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        train_val_idx, test_idx = next(gss_test.split(df, df[label_col], df["patient_id"]))
        
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        val_relative_ratio = val_ratio / (train_ratio + val_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=random_seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, train_val_df[label_col], train_val_df["patient_id"]))

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        # Stratified random split
        # Check if each class has at least 2 samples for stratification
        class_counts = df[label_col].value_counts()
        can_stratify = (class_counts.min() >= 2) if len(class_counts) > 1 else False

        stratify_labels = df[label_col] if can_stratify else None
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=stratify_labels,
        )

        val_relative_ratio = val_ratio / (train_ratio + val_ratio)
        stratify_val = train_val_df[label_col] if can_stratify and (train_val_df[label_col].value_counts().min() >= 2) else None
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_ratio,
            random_state=random_seed,
            stratify=stratify_val,
        )

    # Save split manifests
    prefix = f"{mod_enum.value}_{task_enum.value}"
    train_path = out_dir / f"{prefix}_train.csv"
    val_path = out_dir / f"{prefix}_val.csv"
    test_path = out_dir / f"{prefix}_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


def load_dataset_splits(
    modality: Union[str, Modality],
    task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load pre-existing split CSV files if they exist.
    """
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    task_enum = DiseaseTask.from_str(task) if isinstance(task, str) else task
    s_dir = Path(splits_dir).resolve() if splits_dir else get_splits_dir()

    prefix = f"{mod_enum.value}_{task_enum.value}"
    train_path = s_dir / f"{prefix}_train.csv"
    val_path = s_dir / f"{prefix}_val.csv"
    test_path = s_dir / f"{prefix}_test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
    return None
