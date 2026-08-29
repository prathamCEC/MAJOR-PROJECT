"""
Pydantic Input Validation Schemas for Patient Clinical Health Variables.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PatientClinicalInput(BaseModel):
    """
    Validated clinical variables required by the Phase 6 FT-Transformer.
    """
    patient_id: str = Field(default="PATIENT_NEW_01", description="Unique Patient / Sample Identifier")
    Old_groups: str = Field(default="O_CD", description="Age/Cognitive Group (O_CD, Y_CD, O_CU, Y_CU)")
    Gender: int = Field(default=1, ge=0, le=1, description="Gender (0: Female, 1: Male)")
    Education: float = Field(default=16.0, ge=0.0, le=30.0, description="Years of formal education")
    BMI: float = Field(default=26.5, ge=10.0, le=70.0, description="Body Mass Index (kg/m²)")
    Obese: float = Field(default=0.0, ge=0.0, le=1.0, description="Clinical obesity flag (0 or 1)")
    EtOH_ever: int = Field(default=1, ge=0, le=1, description="History of alcohol consumption (0 or 1)")
    EtOH_current: int = Field(default=0, ge=0, le=1, description="Current alcohol consumption (0 or 1)")
    Smoking_ever: int = Field(default=1, ge=0, le=1, description="History of smoking (0 or 1)")
    Smoking_current: int = Field(default=0, ge=0, le=1, description="Current smoking (0 or 1)")
    HTN: int = Field(default=1, ge=0, le=1, description="Hypertension diagnosed (0: No, 1: Yes)")
    DM2: int = Field(default=0, ge=0, le=1, description="Type 2 Diabetes diagnosed (0: No, 1: Yes)")

    @field_validator("Old_groups")
    @classmethod
    def validate_group(cls, v: str) -> str:
        valid_groups = {"O_CD", "Y_CD", "O_CU", "Y_CU"}
        v_upper = v.upper()
        if v_upper not in valid_groups:
            raise ValueError(f"Old_groups must be one of {valid_groups}, got '{v}'")
        return v_upper

    def to_clinical_dict(self) -> dict:
        """Convert to internal dictionary keyed exactly as expected by Phase 6 schema."""
        return {
            "ID#": self.patient_id,
            "Old groups": self.Old_groups,
            "Gender": self.Gender,
            "Education": self.Education,
            "BMI": self.BMI,
            "Obese": self.Obese,
            "EtOH_ever": self.EtOH_ever,
            "EtOH_current": self.EtOH_current,
            "Smoking_ever": self.Smoking_ever,
            "Smoking_current": self.Smoking_current,
            "HTN": self.HTN,
            "DM2": self.DM2,
        }
