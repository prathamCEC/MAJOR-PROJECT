"""
Tests for Data Leakage detection.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from phase_4_swin_transformer.leakage_check import check_splits_leakage


def test_leakage_clean_splits(tmp_path: Path):
    img_train = tmp_path / "img_train.png"
    img_val = tmp_path / "img_val.png"
    img_test = tmp_path / "img_test.png"

    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 10).save(img_train)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 50).save(img_val)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 100).save(img_test)

    train_df = pd.DataFrame([{"image_path": str(img_train), "patient_id": "P01"}])
    val_df = pd.DataFrame([{"image_path": str(img_val), "patient_id": "P02"}])
    test_df = pd.DataFrame([{"image_path": str(img_test), "patient_id": "P03"}])

    res = check_splits_leakage(train_df, val_df, test_df)
    assert res.passed is True
    assert len(res.overlap_paths) == 0
    assert len(res.patient_overlaps) == 0


def test_leakage_detects_patient_overlap(tmp_path: Path):
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    img3 = tmp_path / "img3.png"

    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 10).save(img1)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 20).save(img2)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 30).save(img3)

    # Same patient P01 in train and test!
    train_df = pd.DataFrame([{"image_path": str(img1), "patient_id": "P01"}])
    val_df = pd.DataFrame([{"image_path": str(img2), "patient_id": "P02"}])
    test_df = pd.DataFrame([{"image_path": str(img3), "patient_id": "P01"}])

    res = check_splits_leakage(train_df, val_df, test_df)
    assert res.passed is False
    assert len(res.patient_overlaps) > 0
