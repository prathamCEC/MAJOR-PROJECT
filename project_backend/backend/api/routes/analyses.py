"""
Analysis Session History and Detail Endpoints.
Allows clinicians to view previous AI evaluation runs and audit outcomes.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...db.models import AnalysisSession, Patient, Prediction, Report, UploadedImage, User, UserRole
from ...db.session import get_db
from ...schemas.analysis_schema import (
    AnalysisSessionDetailResponse,
    AnalysisSessionListItem,
    PredictionSummary,
    UploadedImageSummary,
)
from ..deps import get_current_user

router = APIRouter(prefix="/api/v1/analyses", tags=["Analysis History"])


@router.get("/", response_model=List[AnalysisSessionListItem])
async def list_analysis_sessions(
    patient_code: Optional[str] = Query(None, description="Filter by patient code"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List previous AI analysis sessions for the authenticated clinician."""
    query = (
        select(AnalysisSession)
        .join(Patient, AnalysisSession.patient_id == Patient.id)
        .outerjoin(Prediction, AnalysisSession.id == Prediction.session_id)
        .outerjoin(Report, AnalysisSession.id == Report.session_id)
        .options(
            selectinload(AnalysisSession.patient),
            selectinload(AnalysisSession.prediction),
            selectinload(AnalysisSession.report),
        )
    )

    if current_user.role != UserRole.ADMIN:
        query = query.where(AnalysisSession.user_id == current_user.id)

    if patient_code:
        query = query.where(Patient.patient_code == patient_code.strip().upper())

    query = query.order_by(desc(AnalysisSession.created_at)).offset(skip).limit(limit)
    result = await db.execute(query)
    sessions = result.scalars().all()

    items = []
    for s in sessions:
        pred = s.prediction
        rep = s.report
        items.append(
            AnalysisSessionListItem(
                id=s.id,
                session_uuid=s.session_uuid,
                patient_code=s.patient.patient_code if s.patient else "UNKNOWN",
                status=s.status,
                modalities_requested=s.modalities_requested,
                overall_risk_level=pred.overall_risk_level if pred else None,
                stroke_probability=pred.stroke_probability if pred else None,
                alzheimer_probability=pred.alzheimer_probability if pred else None,
                report_id=rep.report_id if rep else None,
                created_at=s.created_at,
                completed_at=s.completed_at,
            )
        )
    return items


@router.get("/{session_uuid}", response_model=AnalysisSessionDetailResponse)
async def get_analysis_session_detail(
    session_uuid: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full details of a specific analysis session from the database."""
    query = (
        select(AnalysisSession)
        .where(AnalysisSession.session_uuid == session_uuid)
        .options(
            selectinload(AnalysisSession.patient),
            selectinload(AnalysisSession.uploaded_images),
            selectinload(AnalysisSession.prediction),
            selectinload(AnalysisSession.report),
        )
    )

    if current_user.role != UserRole.ADMIN:
        query = query.where(AnalysisSession.user_id == current_user.id)

    result = await db.execute(query)
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{session_uuid}' not found or access denied.",
        )

    images = [
        UploadedImageSummary(
            id=img.id,
            modality=img.modality,
            original_filename=img.original_filename,
            file_size_bytes=img.file_size_bytes,
            quality_score=img.quality_score,
            quality_decision=img.quality_decision,
            created_at=img.created_at,
        )
        for img in s.uploaded_images
    ]

    pred_summary = None
    if s.prediction:
        p = s.prediction
        pred_summary = PredictionSummary(
            stroke_probability=p.stroke_probability,
            stroke_risk_tier=p.stroke_risk_tier,
            stroke_confidence_percent=p.stroke_confidence_percent,
            stroke_variance=p.stroke_variance,
            stroke_entropy=p.stroke_entropy,
            alzheimer_probability=p.alzheimer_probability,
            alzheimer_risk_tier=p.alzheimer_risk_tier,
            alzheimer_confidence_percent=p.alzheimer_confidence_percent,
            alzheimer_variance=p.alzheimer_variance,
            alzheimer_entropy=p.alzheimer_entropy,
            overall_risk_level=p.overall_risk_level,
            created_at=p.created_at,
        )

    return AnalysisSessionDetailResponse(
        id=s.id,
        session_uuid=s.session_uuid,
        patient_id=s.patient_id,
        patient_code=s.patient.patient_code if s.patient else "UNKNOWN",
        user_id=s.user_id,
        status=s.status,
        modalities_requested=s.modalities_requested,
        error_message=s.error_message,
        created_at=s.created_at,
        completed_at=s.completed_at,
        images=images,
        prediction=pred_summary,
        report_id=s.report.report_id if s.report else None,
    )
