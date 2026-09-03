"""
Pydantic Schemas for Patient Management and Clinical Records.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientCreateRequest(BaseModel):
    """Payload to create or register a new patient record."""
    patient_code: str = Field(..., min_length=2, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    full_name: Optional[str] = Field(None, max_length=255)
    age_group: str = Field("O_CD", description="Cohort: O_CD, Y_CD, O_CU, Y_CU")
    gender: int = Field(1, ge=0, le=1, description="1=Male, 0=Female")
    education_years: float = Field(16.0, ge=0.0, le=30.0)
    bmi: float = Field(26.5, ge=10.0, le=70.0)
    obese: float = Field(0.0, ge=0.0, le=1.0)
    hypertension: int = Field(1, ge=0, le=1)
    diabetes_type2: int = Field(0, ge=0, le=1)
    smoking_ever: int = Field(1, ge=0, le=1)
    smoking_current: int = Field(0, ge=0, le=1)
    alcohol_ever: int = Field(1, ge=0, le=1)
    alcohol_current: int = Field(0, ge=0, le=1)


class PatientUpdateRequest(BaseModel):
    """Payload to update an existing patient record."""
    full_name: Optional[str] = None
    age_group: Optional[str] = None
    gender: Optional[int] = None
    education_years: Optional[float] = None
    bmi: Optional[float] = None
    obese: Optional[float] = None
    hypertension: Optional[int] = None
    diabetes_type2: Optional[int] = None
    smoking_ever: Optional[int] = None
    smoking_current: Optional[int] = None
    alcohol_ever: Optional[int] = None
    alcohol_current: Optional[int] = None


class PatientResponse(BaseModel):
    """Patient record response."""
    id: int
    patient_code: str
    owner_user_id: int
    full_name: Optional[str] = None
    age_group: str
    gender: int
    education_years: float
    bmi: float
    obese: float
    hypertension: int
    diabetes_type2: int
    smoking_ever: int
    smoking_current: int
    alcohol_ever: int
    alcohol_current: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
