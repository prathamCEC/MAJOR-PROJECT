"""
Tests for ClinicalSchema.
"""

from pathlib import Path
import pandas as pd
import pytest

from phase_6_clinical_transformer.schema import ClinicalSchema, get_default_retinal_clinical_schema


def test_default_schema_properties():
    schema = get_default_retinal_clinical_schema()
    assert schema.num_numerical == 2
    assert schema.num_categorical == 9
    assert len(schema.all_feature_columns) == 11
    assert schema.patient_id_column == "ID#"


def test_schema_validation_on_dataframe():
    schema = ClinicalSchema(
        numerical_features=["age", "bmi"],
        categorical_features=["sex"],
        binary_features=["smoking"],
        patient_id_column="patient_id",
    )

    valid_df = pd.DataFrame({
        "patient_id": ["P01", "P02"],
        "age": [65, 72],
        "bmi": [24.5, 28.1],
        "sex": ["M", "F"],
        "smoking": [0, 1],
    })
    errors = schema.validate_dataframe(valid_df)
    assert len(errors) == 0

    # Missing column test
    invalid_df = valid_df.drop(columns=["bmi"])
    errors = schema.validate_dataframe(invalid_df)
    assert len(errors) == 1
    assert "bmi" in errors[0]


def test_schema_json_serialization(tmp_path: Path):
    schema = get_default_retinal_clinical_schema()
    json_path = tmp_path / "schema.json"
    schema.save_json(json_path)

    loaded = ClinicalSchema.load_json(json_path)
    assert loaded.numerical_features == schema.numerical_features
    assert loaded.categorical_features == schema.categorical_features
    assert loaded.binary_features == schema.binary_features
