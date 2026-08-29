"""
Report Download & Retrieval Endpoints.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ...core.config import settings
from ...core.logging_config import logger

router = APIRouter(prefix="/api/v1", tags=["Reports"])


@router.get("/report/{report_id}/pdf")
async def download_report_pdf(report_id: str):
    """
    Download compiled Clinical Assessment Report PDF.
    """
    pdf_dir = settings.REPORTS_DIR / "pdf"
    
    # Search for matching report PDF
    matched = list(pdf_dir.glob(f"*{report_id}*.pdf"))
    if not matched:
        # Also check default phase 11 reports directory
        p11_pdf_dir = settings.PROJECT_ROOT / "phase_11_report_generator" / "outputs" / "reports" / "pdf"
        matched = list(p11_pdf_dir.glob(f"*{report_id}*.pdf"))

    if not matched or not matched[0].exists():
        logger.warning(f"Report PDF not found for report_id: {report_id}")
        raise HTTPException(status_code=404, detail=f"PDF report '{report_id}' not found.")

    pdf_file = matched[0]
    return FileResponse(
        path=str(pdf_file),
        filename=pdf_file.name,
        media_type="application/pdf",
    )


@router.get("/report/{report_id}/json")
async def download_report_json(report_id: str):
    """
    Download machine-readable Assessment Report JSON artifact.
    """
    json_dir = settings.REPORTS_DIR / "json"
    
    matched = list(json_dir.glob(f"*{report_id}*.json"))
    if not matched:
        p11_json_dir = settings.PROJECT_ROOT / "phase_11_report_generator" / "outputs" / "reports" / "json"
        matched = list(p11_json_dir.glob(f"*{report_id}*.json"))

    if not matched or not matched[0].exists():
        logger.warning(f"Report JSON not found for report_id: {report_id}")
        raise HTTPException(status_code=404, detail=f"JSON report '{report_id}' not found.")

    json_file = matched[0]
    return FileResponse(
        path=str(json_file),
        filename=json_file.name,
        media_type="application/json",
    )
