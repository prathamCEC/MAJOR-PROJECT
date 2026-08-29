"""
Enumerations and standard definitions for Phase 4 Swin Transformer.

Defines modalities, disease tasks, dataset splits, and prediction structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union


class Modality(str, Enum):
    """Supported imaging modalities for Phase 4."""
    OCTA = "octa"
    OCTB = "octb"
    FUNDUS = "fundus"

    @classmethod
    def from_str(cls, value: Union[str, "Modality"]) -> "Modality":
        if isinstance(value, cls):
            return value
        val_str = value.value if hasattr(value, "value") else str(value)
        clean = val_str.strip().lower()
        for member in cls:
            if member.value == clean or member.name.lower() == clean:
                return member
        raise ValueError(
            f"Unsupported modality '{value}'. Expected one of: {[m.value for m in cls]}"
        )


class DiseaseTask(str, Enum):
    """Clinical analysis tasks."""
    STROKE = "stroke"
    ALZHEIMERS = "alzheimers"
    MULTI_DISEASE = "multi_disease"

    @classmethod
    def from_str(cls, value: Union[str, "DiseaseTask"]) -> "DiseaseTask":
        if isinstance(value, cls):
            return value
        val_str = value.value if hasattr(value, "value") else str(value)
        clean = val_str.strip().lower()
        for member in cls:
            if member.value == clean or member.name.lower() == clean:
                return member
        raise ValueError(
            f"Unsupported disease task '{value}'. Expected one of: {[m.value for m in cls]}"
        )


class SplitType(str, Enum):
    """Dataset partition types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class PredictionOutput:
    """
    Standardized prediction output.
    
    Adheres strictly to research terminology:
    "Predicted Class", "Prediction Probability", and "Model Confidence".
    Does NOT claim clinical diagnosis.
    """
    image_name: str
    image_path: str
    modality: str
    predicted_class: str
    predicted_label: int
    confidence: float
    probabilities: Dict[str, float]
    disclaimer: str = (
        "RESEARCH USE ONLY: This output represents AI model confidence "
        "and is not a clinical medical diagnosis."
    )
