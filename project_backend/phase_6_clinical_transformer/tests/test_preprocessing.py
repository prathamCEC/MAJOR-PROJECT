"""
Tests for ClinicalPreprocessor: numerical standardization, imputation, and category mapping.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from phase_6_clinical_transformer.schema import ClinicalSchema
from phase_6_clinical_transformer.preprocessing import ClinicalPreprocessor


@pytest.fixture
def sample_clinical_df() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["P01", "P02", "P03", "P04"],
        "age": [60.0, 70.0, np.nan, 80.0],
        "bmi": [22.0, 26.0, 30.0, 24.0],
        "gender": ["M", "F", "F", "M"],
        "smoking": [1, 0, 1, 0],
    })


def test_preprocessor_fit_transform(sample_clinical_df: pd.DataFrame):
    schema = ClinicalSchema(
        numerical_features=["age", "bmi"],
        categorical_features=["gender"],
        binary_features=["smoking"],
        patient_id_column="patient_id",
    )
    preprocessor = ClinicalPreprocessor(schema=schema)
    num_mat, cat_mat, pids = preprocessor.fit_transform(sample_clinical_df)

    assert num_mat.shape == (4, 2)
    assert cat_mat.shape == (4, 2)
    assert pids == ["P01", "P02", "P03", "P04"]

    # Check that age NaN was imputed (no NaNs in output)
    assert not np.isnan(num_mat).any()
    # Check median age is 70.0
    assert preprocessor.state.numerical_medians["age"] == 70.0


def test_preprocessor_no_leakage_on_test_data(sample_clinical_df: pd.DataFrame):
    schema = ClinicalSchema(
        numerical_features=["age", "bmi"],
        categorical_features=["gender"],
        binary_features=["smoking"],
        patient_id_column="patient_id",
    )
    preprocessor = ClinicalPreprocessor(schema=schema)
    preprocessor.fit(sample_clinical_df)

    # Test DataFrame with unseen categories and extreme values
    test_df = pd.DataFrame({
        "patient_id": ["P99"],
        "age": [65.0],
        "bmi": [25.0],
        "gender": ["UNKNOWN_VAL"],  # Unseen category
        "smoking": [1],
    })

    num_mat, cat_mat, pids = preprocessor.transform(test_df)
    assert num_mat.shape == (1, 2)
    assert cat_mat.shape == (1, 2)
    # Unseen category mapped to index 0 (<UNK>)
    assert cat_mat[0, 0] == 0


def test_preprocessor_json_serialization(tmp_path: Path, sample_clinical_df: pd.DataFrame):
    schema = ClinicalSchema(
        numerical_features=["age", "bmi"],
        categorical_features=["gender"],
        binary_features=["smoking"],
        patient_id_column="patient_id",
    )
    preprocessor = ClinicalPreprocessor(schema=schema)
    preprocessor.fit(sample_clinical_df)

    save_p = tmp_path / "preprocessor.json"
    preprocessor.save_json(save_p)

    loaded = ClinicalPreprocessor.load_json(save_p)
    assert loaded.state.fitted is True
    assert loaded.state.numerical_medians == preprocessor.state.numerical_medians
    assert loaded.state.category_to_idx == preprocessor.state.category_to_idx
