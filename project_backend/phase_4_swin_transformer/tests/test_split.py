"""
Tests for dataset splitting.
"""

from pathlib import Path
import pandas as pd
import pytest

from phase_4_swin_transformer.enums import DiseaseTask, Modality
from phase_4_swin_transformer.split_dataset import create_dataset_splits


def test_create_dataset_splits_stratified(tmp_path: Path):
    data = []
    for i in range(20):
        data.append({
            "image_path": f"/path/to/img_{i}.png",
            "modality": "octa",
            "label": i % 2,
            "class_name": "normal" if i % 2 == 0 else "disease",
            "patient_id": f"P{i:02d}",
        })
    df = pd.DataFrame(data)

    train_df, val_df, test_df = create_dataset_splits(
        df,
        modality=Modality.OCTA,
        task=DiseaseTask.ALZHEIMERS,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir=tmp_path,
    )

    assert len(train_df) + len(val_df) + len(test_df) == 20
    assert len(train_df) >= 12
    assert len(val_df) >= 2
    assert len(test_df) >= 2

    # Check manifest files exist
    assert (tmp_path / "octa_alzheimers_train.csv").exists()
    assert (tmp_path / "octa_alzheimers_val.csv").exists()
    assert (tmp_path / "octa_alzheimers_test.csv").exists()
