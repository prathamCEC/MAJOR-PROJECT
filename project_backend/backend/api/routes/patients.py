"""
Patient Management Endpoints: Create, List, View, Update, and Delete.
Enforces ownership authorization (users can only access their own patients).
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Patient, User, UserRole
from ...db.session import get_db
from ...schemas.auth_schema import MessageResponse
from ...schemas.patient_schema import (
    PatientCreateRequest,
    PatientResponse,
    PatientUpdateRequest,
)
from ..deps import get_current_user, record_audit_log

router = APIRouter(prefix="/api/v1/patients", tags=["Patient Management"])


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    req: PatientCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register a new patient profile associated with the authenticated clinician."""
    code_clean = req.patient_code.strip().upper()

    # Check for duplicate patient code
    existing = await db.execute(select(Patient).where(Patient.patient_code == code_clean))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient with code '{code_clean}' already exists.",
        )

    new_patient = Patient(
        patient_code=code_clean,
        owner_user_id=current_user.id,
        full_name=req.full_name,
        age_group=req.age_group,
        gender=req.gender,
        education_years=req.education_years,
        bmi=req.bmi,
        obese=req.obese,
        hypertension=req.hypertension,
        diabetes_type2=req.diabetes_type2,
        smoking_ever=req.smoking_ever,
        smoking_current=req.smoking_current,
        alcohol_ever=req.alcohol_ever,
        alcohol_current=req.alcohol_current,
    )
    db.add(new_patient)
    await db.commit()
    await db.refresh(new_patient)

    # Audit Log
    client_ip = request.client.host if request.client else None
    await record_audit_log(
        db=db,
        action="PATIENT_CREATED",
        user_id=current_user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
        details={"patient_code": code_clean},
    )

    return new_patient


@router.get("/", response_model=List[PatientResponse])
async def list_patients(
    search: Optional[str] = Query(None, description="Search by code or name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List patient records.
    Standard users see only their own patients; Admins can see all.
    """
    query = select(Patient)
    if current_user.role != UserRole.ADMIN:
        query = query.where(Patient.owner_user_id == current_user.id)

    if search:
        s_term = f"%{search.strip()}%"
        query = query.where(
            or_(
                Patient.patient_code.ilike(s_term),
                Patient.full_name.ilike(s_term),
            )
        )

    query = query.order_by(Patient.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{patient_code}", response_model=PatientResponse)
async def get_patient(
    patient_code: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve an individual patient record with ownership check."""
    code_clean = patient_code.strip().upper()
    query = select(Patient).where(Patient.patient_code == code_clean)
    if current_user.role != UserRole.ADMIN:
        query = query.where(Patient.owner_user_id == current_user.id)

    result = await db.execute(query)
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{code_clean}' not found or you do not have permission to access it.",
        )
    return patient


@router.put("/{patient_code}", response_model=PatientResponse)
async def update_patient(
    patient_code: str,
    req: PatientUpdateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update patient clinical variables with ownership check."""
    code_clean = patient_code.strip().upper()
    query = select(Patient).where(Patient.patient_code == code_clean)
    if current_user.role != UserRole.ADMIN:
        query = query.where(Patient.owner_user_id == current_user.id)

    result = await db.execute(query)
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient '{code_clean}' not found.")

    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(patient, key, value)

    await db.commit()
    await db.refresh(patient)

    # Audit Log
    client_ip = request.client.host if request.client else None
    await record_audit_log(
        db=db,
        action="PATIENT_UPDATED",
        user_id=current_user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
        details={"patient_code": code_clean, "updated_fields": list(update_data.keys())},
    )

    return patient


@router.delete("/{patient_code}", response_model=MessageResponse)
async def delete_patient(
    patient_code: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a patient record and cascade sessions."""
    code_clean = patient_code.strip().upper()
    query = select(Patient).where(Patient.patient_code == code_clean)
    if current_user.role != UserRole.ADMIN:
        query = query.where(Patient.owner_user_id == current_user.id)

    result = await db.execute(query)
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient '{code_clean}' not found.")

    await db.delete(patient)
    await db.commit()

    # Audit Log
    client_ip = request.client.host if request.client else None
    await record_audit_log(
        db=db,
        action="PATIENT_DELETED",
        user_id=current_user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
        details={"patient_code": code_clean},
    )

    return MessageResponse(message=f"Patient '{code_clean}' deleted successfully.")
