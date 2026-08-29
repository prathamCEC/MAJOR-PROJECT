"""
Tests for dataset validation and label verification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from phase_4_swin_transformer.enums import DiseaseTask, Modality
from phase_4_swin_transformer.validation import DatasetValidator


def test_validator_with_valid_csv(tmp_path: Path):
    # Create sample valid images
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 100).save(img1)
    Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 150).save(img2)

    csv_p = tmp_path / "manifest.csv"
    df = pd.DataFrame([
        {"image_path": str(img1), "modality": "octa", "label": 0, "class_name": "normal", "patient_id": "P01"},
        {"image_path": str(img2), "modality": "octa", "label": 1, "class_name": "disease", "patient_id": "P02"},
    ])
    df.to_csv(csv_p, index=False)

    validator = DatasetValidator(task=DiseaseTask.ALZHEIMERS)
    stats = validator.validate_csv_manifest(csv_p, modality=Modality.OCTA)

    assert stats.total_images == 2
    assert stats.valid_images == 2
    assert stats.has_verified_labels is True
    assert stats.has_patient_ids is True
    assert stats.patient_count == 2


def test_validator_detects_missing_verified_labels(tmp_path: Path):
    # Create dataset with only 1 class (cannot perform binary classification)
    img1 = tmp_path / "img1.png"
    Image.fromarray(np.ones((64, 64), dtype=np.uint8)).save(img1)

    csv_p = tmp_path / "single_class.csv"
    df = pd.DataFrame([
        {"image_path": str(img1), "modality": "octa", "label": 0, "class_name": "normal"},
    ])
    df.to_csv(csv_p, index=False)

    validator = DatasetValidator()
    stats = validator.validate_csv_manifest(csv_p, modality=Modality.OCTA)
    assert stats.has_verified_labels is False
    assert len(stats.error_messages) > 0
