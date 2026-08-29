"""
Clinical Schema System for Phase 6 FT-Transformer.

Defines structural specification, feature types (numerical, categorical, binary),
metadata/patient identifier handling, and imputation strategies for structured clinical data.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


@dataclass
class ClinicalSchema:
    """
    Schema configuration for patient tabular clinical variables.
    """
    # Feature columns by statistical type
    numerical_features: List[str] = field(default_factory=lambda: ["BMI", "Education"])
    categorical_features: List[str] = field(default_factory=lambda: ["Old groups", "Gender", "Obese"])
    binary_features: List[str] = field(default_factory=lambda: [
        "EtOH_ever", "EtOH_current", "Smoking_ever", "Smoking_current", "HTN", "DM2"
    ])

    # Metadata and identifier columns (excluded from model input representations)
    patient_id_column: Optional[str] = "ID#"
    label_columns: List[str] = field(default_factory=lambda: ["AD"])

    # Imputation strategies
    numerical_imputation: str = "median"  # 'median', 'mean', 'zero'
    categorical_imputation: str = "unknown"  # 'unknown', 'most_frequent'

    # Normalization method for continuous numerical features
    numerical_scaling: str = "standard"  # 'standard', 'robust', 'minmax'

    @property
    def all_feature_columns(self) -> List[str]:
        """Return ordered list of all input features (numerical + categorical + binary)."""
        return self.numerical_features + self.categorical_features + self.binary_features

    @property
    def num_numerical(self) -> int:
        return len(self.numerical_features)

    @property
    def num_categorical(self) -> int:
        # Binary features are treated as 2-category categorical embeddings
        return len(self.categorical_features) + len(self.binary_features)

    @property
    def all_categorical_like(self) -> List[str]:
        return self.categorical_features + self.binary_features

    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """
        Validate that input DataFrame contains all specified schema columns.
        Returns list of error messages (empty if valid).
        """
        errors = []
        if df.empty:
            errors.append("Input clinical DataFrame is empty.")
            return errors

        missing_cols = [col for col in self.all_feature_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required clinical feature columns: {missing_cols}")

        if self.patient_id_column and self.patient_id_column not in df.columns:
            errors.append(f"Patient ID column '{self.patient_id_column}' not found in dataset.")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalSchema":
        return cls(**data)

    def save_json(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "ClinicalSchema":
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_retinal_clinical_schema() -> ClinicalSchema:
    """
    Returns the standard default schema matching the real clinical dataset (5_ASSOCIATED DATA.xlsx).
    """
    return ClinicalSchema(
        numerical_features=["BMI", "Education"],
        categorical_features=["Old groups", "Gender", "Obese"],
        binary_features=["EtOH_ever", "EtOH_current", "Smoking_ever", "Smoking_current", "HTN", "DM2"],
        patient_id_column="ID#",
        label_columns=["AD"],
        numerical_imputation="median",
        categorical_imputation="unknown",
        numerical_scaling="standard",
    )
