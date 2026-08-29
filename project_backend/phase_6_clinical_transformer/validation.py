"""
Data Validation and Quality Auditing Module for Clinical Data.

Performs schema verification, missing value checks, data type validations,
duplicate patient audits, and representation integrity checks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from .schema import ClinicalSchema


@dataclass
class ClinicalAuditReport:
    """Clinical data audit summary."""
    total_records: int = 0
    unique_patients: int = 0
    has_duplicate_patients: bool = False
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = False


class ClinicalDataValidator:
    """
    Validates clinical tabular datasets prior to preprocessing and transformer modeling.
    """

    def __init__(self, schema: ClinicalSchema):
        self.schema = schema

    def audit_dataframe(self, df: pd.DataFrame) -> ClinicalAuditReport:
        """
        Run full audit on clinical DataFrame.
        """
        report = ClinicalAuditReport()
        report.total_records = len(df)

        if df.empty:
            report.errors.append("Clinical dataset is completely empty.")
            return report

        # 1. Schema Column Validation
        schema_errors = self.schema.validate_dataframe(df)
        report.errors.extend(schema_errors)

        # 2. Patient Identifier Audits
        if self.schema.patient_id_column and self.schema.patient_id_column in df.columns:
            patient_series = df[self.schema.patient_id_column].astype(str)
            report.unique_patients = int(patient_series.nunique())
            if report.unique_patients < report.total_records:
                report.has_duplicate_patients = True
                dups = patient_series[patient_series.duplicated()].unique().tolist()
                report.warnings.append(
                    f"Found duplicate patient records for IDs: {dups[:5]} (Total duplicates: {len(dups)})"
                )

        # 3. Missing Value Audits
        for col in self.schema.all_feature_columns:
            if col in df.columns:
                null_count = int(df[col].isnull().sum())
                report.missing_value_counts[col] = null_count
                if null_count > 0:
                    pct = (null_count / report.total_records) * 100
                    report.warnings.append(
                        f"Feature '{col}' contains {null_count} missing values ({pct:.1f}%). "
                        f"Imputation will be applied."
                    )

        # 4. Numerical Feature Data Type Checks
        for col in self.schema.numerical_features:
            if col in df.columns:
                non_numeric = pd.to_numeric(df[col], errors="coerce").isnull() & df[col].notnull()
                invalid_count = int(non_numeric.sum())
                if invalid_count > 0:
                    report.errors.append(
                        f"Numerical feature '{col}' contains {invalid_count} non-numeric string values."
                    )

        report.numerical_features = self.schema.numerical_features
        report.categorical_features = self.schema.all_categorical_like
        report.is_valid = len(report.errors) == 0
        return report


def validate_clinical_representation_output(
    output_dict: Dict[str, torch.Tensor],
    expected_batch_size: int,
    expected_dim: int = 512,
) -> None:
    """
    Validate output from ClinicalFTTransformerModel.
    """
    if "clinical_representation" not in output_dict:
        raise KeyError("Output dictionary does not contain 'clinical_representation' key.")

    cr = output_dict["clinical_representation"]
    if not isinstance(cr, torch.Tensor):
        raise TypeError(f"Clinical representation must be a torch.Tensor, got {type(cr)}.")

    if cr.shape != (expected_batch_size, expected_dim):
        raise ValueError(
            f"Expected Clinical Representation shape ({expected_batch_size}, {expected_dim}), "
            f"but received {tuple(cr.shape)}."
        )

    if torch.isnan(cr).any():
        raise ValueError("Clinical Representation contains NaN values.")

    if torch.isinf(cr).any():
        raise ValueError("Clinical Representation contains Inf values.")
