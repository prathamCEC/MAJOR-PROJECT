"""
Report Download & Retrieval Endpoints with Authorization and Audit Logging.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import settings
from ...core.logging_config import logger
from ...db.models import Report, User, UserRole
from ...db.session import get_db
from ..deps import get_current_user_optional, record_audit_log

router = APIRouter(prefix="/api/v1", tags=["Reports"])


@router.get("/report/{report_id}/pdf")
async def download_report_pdf(
    report_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """
    Download compiled Clinical Assessment Report PDF.
    Enforces authorization if authenticated and records audit log.
    """
    clean_id = report_id.strip()

    # If user is authenticated, check report ownership in database
    if current_user and current_user.role != UserRole.ADMIN:
        rep_res = await db.execute(select(Report).where(Report.report_id == clean_id))
        report_record = rep_res.scalar_one_or_none()
        if report_record and report_record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to download this report.",
            )

    pdf_dir = settings.REPORTS_DIR / "pdf"
    matched = list(pdf_dir.glob(f"*{clean_id}*.pdf"))
    if not matched:
        p11_pdf_dir = settings.PROJECT_ROOT / "phase_11_report_generator" / "outputs" / "reports" / "pdf"
        matched = list(p11_pdf_dir.glob(f"*{clean_id}*.pdf"))

    if not matched or not matched[0].exists():
        logger.warning(f"Report PDF not found for report_id: {clean_id}")
        raise HTTPException(status_code=404, detail=f"PDF report '{clean_id}' not found.")

    pdf_file = matched[0]

    # Audit log
    client_ip = request.client.host if request.client else None
    await record_audit_log(
        db=db,
        action="DOWNLOAD_REPORT_PDF",
        user_id=current_user.id if current_user else None,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
        details={"report_id": clean_id, "file_name": pdf_file.name},
    )

    return FileResponse(
        path=str(pdf_file),
        filename=pdf_file.name,
        media_type="application/pdf",
    )


@router.get("/report/{report_id}/json")
async def download_report_json(
    report_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """
    Download machine-readable Assessment Report JSON artifact.
    """
    clean_id = report_id.strip()

    if current_user and current_user.role != UserRole.ADMIN:
        rep_res = await db.execute(select(Report).where(Report.report_id == clean_id))
        report_record = rep_res.scalar_one_or_none()
        if report_record and report_record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to download this report.",
            )

    json_dir = settings.REPORTS_DIR / "json"
    matched = list(json_dir.glob(f"*{clean_id}*.json"))
    if not matched:
        p11_json_dir = settings.PROJECT_ROOT / "phase_11_report_generator" / "outputs" / "reports" / "json"
        matched = list(p11_json_dir.glob(f"*{clean_id}*.json"))

    if not matched or not matched[0].exists():
        logger.warning(f"Report JSON not found for report_id: {clean_id}")
        raise HTTPException(status_code=404, detail=f"JSON report '{clean_id}' not found.")

    json_file = matched[0]

    client_ip = request.client.host if request.client else None
    await record_audit_log(
        db=db,
        action="DOWNLOAD_REPORT_JSON",
        user_id=current_user.id if current_user else None,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
        details={"report_id": clean_id, "file_name": json_file.name},
    )

    return FileResponse(
        path=str(json_file),
        filename=json_file.name,
        media_type="application/json",
    )


@router.get("/reports")
async def list_reports(
    skip: int = 0,
    limit: int = 50,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """
    List clinical reports. If user is authenticated, filters by user unless admin.
    """
    from sqlalchemy.orm import selectinload
    from ...db.models import Patient, AnalysisSession, Prediction

    query = (
        select(Report)
        .join(Patient, Report.patient_id == Patient.id)
        .options(
            selectinload(Report.patient),
            selectinload(Report.session).selectinload(AnalysisSession.prediction),
        )
        .order_by(Report.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    if current_user and current_user.role != UserRole.ADMIN:
        query = query.where(Report.user_id == current_user.id)

    result = await db.execute(query)
    reports = result.scalars().all()

    report_list = []
    for r in reports:
        pred = r.session.prediction if r.session else None
        report_list.append({
            "id": r.id,
            "report_id": r.report_id,
            "patient_code": r.patient.patient_code if r.patient else "UNKNOWN",
            "patient_name": r.patient.full_name if r.patient else None,
            "overall_risk_level": pred.overall_risk_level.value if pred else "UNKNOWN",
            "stroke_probability": pred.stroke_probability if pred else None,
            "alzheimer_probability": pred.alzheimer_probability if pred else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "pdf_url": f"/api/v1/report/{r.report_id}/pdf",
            "json_url": f"/api/v1/report/{r.report_id}/json",
        })

    return report_list

