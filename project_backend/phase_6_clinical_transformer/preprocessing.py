"""
Clinical Data Preprocessing and Feature Encoding Module.

Implements robust, leakage-free tabular preprocessing for numerical and categorical
clinical features. Preprocessing statistics are fitted strictly on the training split.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .schema import ClinicalSchema, get_default_retinal_clinical_schema


@dataclass
class PreprocessorState:
    """Serializable fitted statistics."""
    numerical_medians: Dict[str, float] = field(default_factory=dict)
    numerical_means: Dict[str, float] = field(default_factory=dict)
    numerical_stds: Dict[str, float] = field(default_factory=dict)
    category_to_idx: Dict[str, Dict[str, int]] = field(default_factory=dict)
    category_cardinalities: Dict[str, int] = field(default_factory=dict)
    fitted: bool = False


class ClinicalPreprocessor:
    """
    Fits and transforms clinical tabular DataFrames into normalized numerical tensors
    and categorical index arrays.
    """

    def __init__(self, schema: Optional[ClinicalSchema] = None):
        self.schema = schema or get_default_retinal_clinical_schema()
        self.state = PreprocessorState()

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        """
        Fit preprocessing statistics (medians, means, standard deviations, category mappings)
        strictly on the provided (training) dataset.
        """
        errors = self.schema.validate_dataframe(df)
        if errors:
            raise ValueError(f"Schema validation failed during preprocessor fit: {errors}")

        # 1. Fit Numerical Features
        for col in self.schema.numerical_features:
            series = pd.to_numeric(df[col], errors="coerce")
            valid_vals = series.dropna()

            if len(valid_vals) == 0:
                med_val = 0.0
                mean_val = 0.0
                std_val = 1.0
            else:
                med_val = float(valid_vals.median())
                mean_val = float(valid_vals.mean())
                std_val = float(valid_vals.std()) if len(valid_vals) > 1 and valid_vals.std() > 0 else 1.0

            self.state.numerical_medians[col] = med_val
            self.state.numerical_means[col] = mean_val
            self.state.numerical_stds[col] = std_val

        # 2. Fit Categorical and Binary Features
        # Index 0 is always reserved for '<UNK>' (missing / unknown categories)
        for col in self.schema.all_categorical_like:
            series = df[col].astype(str).str.strip()
            # Exclude missing representations from vocabulary
            clean_series = series[~series.isin(["nan", "None", "", "NaN", "null"])]
            unique_cats = sorted(list(clean_series.unique()))

            cat_map = {"<UNK>": 0}
            for idx, cat_val in enumerate(unique_cats, start=1):
                cat_map[cat_val] = idx

            self.state.category_to_idx[col] = cat_map
            # Cardinality includes <UNK> (len(cat_map))
            self.state.category_cardinalities[col] = len(cat_map)

        self.state.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transform raw clinical DataFrame into normalized numerical array and categorical index array.

        Args:
            df: Clinical DataFrame.

        Returns:
            Tuple of:
            - numerical_matrix: Float32 array [N, num_numerical]
            - categorical_matrix: Int64 array [N, num_categorical]
            - patient_ids: List of patient identifier strings (or index strings if not provided)
        """
        if not self.state.fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted on training data before calling transform().")

        errors = self.schema.validate_dataframe(df)
        if errors:
            raise ValueError(f"Schema validation failed during transform: {errors}")

        n_samples = len(df)

        # 1. Transform Numerical Features
        if self.schema.numerical_features:
            num_cols = []
            for col in self.schema.numerical_features:
                series = pd.to_numeric(df[col], errors="coerce").copy()
                # Impute missing values with fitted training median
                fill_val = self.state.numerical_medians[col]
                series = series.fillna(fill_val)

                # Standardize using fitted training statistics
                mean_val = self.state.numerical_means[col]
                std_val = self.state.numerical_stds[col]
                standardized = (series.values - mean_val) / std_val
                num_cols.append(standardized)

            numerical_matrix = np.column_stack(num_cols).astype(np.float32)
        else:
            numerical_matrix = np.empty((n_samples, 0), dtype=np.float32)

        # 2. Transform Categorical and Binary Features
        if self.schema.all_categorical_like:
            cat_cols = []
            for col in self.schema.all_categorical_like:
                series = df[col].astype(str).str.strip()
                cat_map = self.state.category_to_idx[col]

                # Map each entry to integer index; unknown/missing map to 0 (<UNK>)
                mapped = [cat_map.get(val, 0) for val in series]
                cat_cols.append(mapped)

            categorical_matrix = np.column_stack(cat_cols).astype(np.int64)
        else:
            categorical_matrix = np.empty((n_samples, 0), dtype=np.int64)

        # 3. Extract Patient Identifiers (Metadata only)
        if self.schema.patient_id_column and self.schema.patient_id_column in df.columns:
            patient_ids = df[self.schema.patient_id_column].astype(str).tolist()
        else:
            patient_ids = [f"patient_{i}" for i in range(n_samples)]

        return numerical_matrix, categorical_matrix, patient_ids

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit on DataFrame and transform in one step."""
        return self.fit(df).transform(df)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessor state and schema."""
        return {
            "schema": self.schema.to_dict(),
            "state": {
                "numerical_medians": self.state.numerical_medians,
                "numerical_means": self.state.numerical_means,
                "numerical_stds": self.state.numerical_stds,
                "category_to_idx": self.state.category_to_idx,
                "category_cardinalities": self.state.category_cardinalities,
                "fitted": self.state.fitted,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalPreprocessor":
        """Reconstruct preprocessor from dictionary."""
        schema = ClinicalSchema.from_dict(data["schema"])
        preprocessor = cls(schema=schema)

        st = data["state"]
        preprocessor.state = PreprocessorState(
            numerical_medians=st["numerical_medians"],
            numerical_means=st["numerical_means"],
            numerical_stds=st["numerical_stds"],
            category_to_idx=st["category_to_idx"],
            category_cardinalities=st["category_cardinalities"],
            fitted=st["fitted"],
        )
        return preprocessor

    def save_json(self, file_path: Union[str, Path]) -> Path:
        """Save preprocessor state to JSON."""
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "ClinicalPreprocessor":
        """Load preprocessor from JSON file."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
