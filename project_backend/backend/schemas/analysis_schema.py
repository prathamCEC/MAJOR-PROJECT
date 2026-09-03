"""
Pydantic Schemas for Analysis History and Session Status.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from ..db.models import AnalysisStatus, ModalityEnum, RiskTierEnum


class UploadedImageSummary(BaseModel):
    """Uploaded retinal scan summary."""
    id: int
    modality: ModalityEnum
    original_filename: str
    file_size_bytes: int
    quality_score: Optional[float] = None
    quality_decision: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionSummary(BaseModel):
    """Prediction summary stored in database."""
    stroke_probability: float
    stroke_risk_tier: RiskTierEnum
    stroke_confidence_percent: float
    stroke_variance: float
    stroke_entropy: float
    alzheimer_probability: float
    alzheimer_risk_tier: RiskTierEnum
    alzheimer_confidence_percent: float
    alzheimer_variance: float
    alzheimer_entropy: float
    overall_risk_level: RiskTierEnum
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AnalysisSessionDetailResponse(BaseModel):
    """Detailed view of an analysis session from the database."""
    id: int
    session_uuid: str
    patient_id: int
    patient_code: str
    user_id: int
    status: AnalysisStatus
    modalities_requested: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    images: List[UploadedImageSummary] = []
    prediction: Optional[PredictionSummary] = None
    report_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AnalysisSessionListItem(BaseModel):
    """Compact summary item for analysis history table."""
    id: int
    session_uuid: str
    patient_code: str
    status: AnalysisStatus
    modalities_requested: str
    overall_risk_level: Optional[RiskTierEnum] = None
    stroke_probability: Optional[float] = None
    alzheimer_probability: Optional[float] = None
    report_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
