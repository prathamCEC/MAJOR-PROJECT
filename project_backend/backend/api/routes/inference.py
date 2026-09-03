"""
Patient Diagnostic Inference Endpoints with Database Persistence and Authentication.
"""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import settings
from ...core.logging_config import logger
from ...db.models import (
    AnalysisSession,
    AnalysisStatus,
    ModalityEnum,
    Patient,
    Prediction,
    Report,
    RiskTierEnum,
    UploadedImage,
    User,
    UserRole,
)
from ...db.session import get_db
from ...schemas.input_schema import PatientClinicalInput
from ...schemas.output_schema import AnalysisResponse
from ...services.inference_service import InferenceService
from ..deps import get_current_user_optional, record_audit_log

router = APIRouter(prefix="/api/v1", tags=["Multimodal Analysis"])


def validate_image_payload(
    file_bytes: bytes,
    filename: str,
    modality_name: str,
) -> tuple[int, int, str]:
    """
    Validate image file size, magic bytes signature, and dimensions.
    Returns (width, height, sha256_hash).
    """
    # 1. Size Check
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{modality_name.upper()} image exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
        )

    # 2. Signature and Readability Check via PIL
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img.verify()
            width, height = img.size
            img_format = img.format.lower() if img.format else "png"
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or invalid {modality_name.upper()} image file. Security signature verification failed.",
        )

    if width < 32 or height < 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{modality_name.upper()} image resolution ({width}x{height}) is too small for clinical AI processing.",
        )

    sha256_hash = hashlib.sha256(file_bytes).hexdigest()
    return width, height, sha256_hash


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient_endpoint(
    request: Request,
    patient_id: str = Form("PATIENT_NEW_01"),
    Old_groups: str = Form("O_CD"),
    Gender: int = Form(1),
    Education: float = Form(16.0),
    BMI: float = Form(26.5),
    Obese: float = Form(0.0),
    EtOH_ever: int = Form(1),
    EtOH_current: int = Form(0),
    Smoking_ever: int = Form(1),
    Smoking_current: int = Form(0),
    HTN: int = Form(1),
    DM2: int = Form(0),
    octa_file: Optional[UploadFile] = File(None),
    octb_file: Optional[UploadFile] = File(None),
    fundus_file: Optional[UploadFile] = File(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """
    Execute full multimodal inference (Phase 2 to Phase 11) for a patient.
    Persists analysis session, uploaded images, predictions, and reports to SQL database.
    """
    # 1. Validate Clinical Form Input
    try:
        clinical_input = PatientClinicalInput(
            patient_id=patient_id,
            Old_groups=Old_groups,
            Gender=Gender,
            Education=Education,
            BMI=BMI,
            Obese=Obese,
            EtOH_ever=EtOH_ever,
            EtOH_current=EtOH_current,
            Smoking_ever=Smoking_ever,
            Smoking_current=Smoking_current,
            HTN=HTN,
            DM2=DM2,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid clinical inputs: {str(e)}")

    # 2. Read Image Bytes & Validate Signatures
    octa_bytes = await octa_file.read() if octa_file else None
    octb_bytes = await octb_file.read() if octb_file else None
    fundus_bytes = await fundus_file.read() if fundus_file else None

    # Validate that at least one modality is supplied
    has_image = any(b is not None and len(b) > 0 for b in (octa_bytes, octb_bytes, fundus_bytes))
    if not has_image:
        raise HTTPException(
            status_code=422,
            detail="Please select and provide at least one retinal imaging modality (OCT-A, OCT-B, or Fundus).",
        )

    # Security validation on uploaded bytes
    image_meta = {}
    if octa_bytes and len(octa_bytes) > 0:
        w, h, hsh = validate_image_payload(octa_bytes, octa_file.filename, "octa")
        image_meta["octa"] = {"w": w, "h": h, "hash": hsh, "name": octa_file.filename, "size": len(octa_bytes), "mime": octa_file.content_type or "image/png"}
    if octb_bytes and len(octb_bytes) > 0:
        w, h, hsh = validate_image_payload(octb_bytes, octb_file.filename, "octb")
        image_meta["octb"] = {"w": w, "h": h, "hash": hsh, "name": octb_file.filename, "size": len(octb_bytes), "mime": octb_file.content_type or "image/png"}
    if fundus_bytes and len(fundus_bytes) > 0:
        w, h, hsh = validate_image_payload(fundus_bytes, fundus_file.filename, "fundus")
        image_meta["fundus"] = {"w": w, "h": h, "hash": hsh, "name": fundus_file.filename, "size": len(fundus_bytes), "mime": fundus_file.content_type or "image/png"}

    # 3. Resolve User Context (Use authenticated user or fallback to system admin)
    user_id = current_user.id if current_user else None
    if not user_id:
        admin_res = await db.execute(select(User).where(User.role == UserRole.ADMIN).limit(1))
        first_admin = admin_res.scalar_one_or_none()
        user_id = first_admin.id if first_admin else 1

    # 4. Resolve or Create Patient Record
    code_clean = patient_id.strip().upper()
    patient_res = await db.execute(select(Patient).where(Patient.patient_code == code_clean))
    patient_record = patient_res.scalar_one_or_none()

    if not patient_record:
        patient_record = Patient(
            patient_code=code_clean,
            owner_user_id=user_id,
            full_name=f"Patient {code_clean}",
            age_group=Old_groups,
            gender=Gender,
            education_years=Education,
            bmi=BMI,
            obese=Obese,
            hypertension=HTN,
            diabetes_type2=DM2,
            smoking_ever=Smoking_ever,
            smoking_current=Smoking_current,
            alcohol_ever=EtOH_ever,
            alcohol_current=EtOH_current,
        )
        db.add(patient_record)
        await db.flush()

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Audit Log: IMAGE_UPLOAD
    await record_audit_log(
        db=db,
        action="IMAGE_UPLOAD",
        user_id=user_id,
        ip_address=client_ip,
        user_agent=user_agent,
        details={"modalities": list(image_meta.keys()), "patient_code": code_clean},
    )

    # Audit Log: ANALYSIS_STARTED
    await record_audit_log(
        db=db,
        action="ANALYSIS_STARTED",
        user_id=user_id,
        ip_address=client_ip,
        user_agent=user_agent,
        details={"patient_code": code_clean, "modalities": list(image_meta.keys())},
    )

    # 5. Execute Core Multimodal AI Pipeline
    try:
        service = InferenceService()
        response: AnalysisResponse = service.analyze_patient(
            clinical_input=clinical_input,
            octa_bytes=octa_bytes,
            octb_bytes=octb_bytes,
            fundus_bytes=fundus_bytes,
        )
    except ValueError as e:
        await record_audit_log(
            db=db,
            action="ANALYSIS_FAILED",
            user_id=user_id,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"patient_code": code_clean, "error": str(e)},
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference execution failure: {e}", exc_info=True)
        await record_audit_log(
            db=db,
            action="ANALYSIS_FAILED",
            user_id=user_id,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"patient_code": code_clean, "error": "Internal pipeline execution error"},
        )
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")

    # 6. Database Transaction: Persist Analysis Session, Images, Predictions, and Report
    try:
        modalities_str = ",".join(response.modalities_processed)
        session_record = AnalysisSession(
            session_uuid=response.session_id,
            patient_id=patient_record.id,
            user_id=user_id,
            status=AnalysisStatus.COMPLETED,
            modalities_requested=modalities_str,
            completed_at=datetime.now(timezone.utc),
        )
        db.add(session_record)
        await db.flush()

        # Save Uploaded Images metadata
        for m_name, meta in image_meta.items():
            q_score = None
            q_dec = None
            if m_name in response.image_quality:
                q_score = response.image_quality[m_name].quality_score
                q_dec = response.image_quality[m_name].decision

            img_rec = UploadedImage(
                session_id=session_record.id,
                patient_id=patient_record.id,
                modality=ModalityEnum(m_name),
                original_filename=meta["name"],
                storage_path=str(settings.UPLOAD_DIR / response.session_id / f"{m_name}_raw.png"),
                mime_type=meta["mime"],
                file_size_bytes=meta["size"],
                image_width=meta["w"],
                image_height=meta["h"],
                sha256_hash=meta["hash"],
                quality_score=q_score,
                quality_decision=q_dec,
            )
            db.add(img_rec)

        # Save Prediction Record
        shap_json_str = json.dumps(response.explainability.get("shap_clinical", {}))
        gradcam_json_str = json.dumps({
            "stroke": response.explainability.get("stroke", {}).get("gradcam", {}),
            "alzheimer": response.explainability.get("alzheimer", {}).get("gradcam", {}),
        })

        def parse_risk_tier(val: str) -> RiskTierEnum:
            upper = str(val).upper()
            if "HIGH" in upper:
                return RiskTierEnum.HIGH
            elif "MOD" in upper:
                return RiskTierEnum.MODERATE
            return RiskTierEnum.LOW

        pred_rec = Prediction(
            session_id=session_record.id,
            patient_id=patient_record.id,
            stroke_probability=response.stroke_prediction.probability,
            stroke_risk_tier=parse_risk_tier(response.stroke_prediction.risk_category),
            stroke_confidence_percent=response.stroke_uncertainty.confidence_percent,
            stroke_variance=response.stroke_uncertainty.predictive_variance,
            stroke_entropy=response.stroke_uncertainty.predictive_entropy,
            alzheimer_probability=response.alzheimer_prediction.probability,
            alzheimer_risk_tier=parse_risk_tier(response.alzheimer_prediction.risk_category),
            alzheimer_confidence_percent=response.alzheimer_uncertainty.confidence_percent,
            alzheimer_variance=response.alzheimer_uncertainty.predictive_variance,
            alzheimer_entropy=response.alzheimer_uncertainty.predictive_entropy,
            overall_risk_level=parse_risk_tier(response.overall_risk_level),
            shap_summary_json=shap_json_str,
            gradcam_paths_json=gradcam_json_str,
        )
        db.add(pred_rec)

        # Save Report Record
        pdf_path_str = response.pdf_report_path
        json_path_str = response.json_report_path
        rep_rec = Report(
            report_id=response.report_id,
            session_id=session_record.id,
            patient_id=patient_record.id,
            user_id=user_id,
            pdf_path=pdf_path_str,
            json_path=json_path_str,
        )
        db.add(rep_rec)

        # Audit Log: ANALYSIS_COMPLETED
        await record_audit_log(
            db=db,
            action="ANALYSIS_COMPLETED",
            user_id=user_id,
            ip_address=client_ip,
            user_agent=user_agent,
            details={
                "patient_code": code_clean,
                "session_id": response.session_id,
                "report_id": response.report_id,
                "overall_risk_level": response.overall_risk_level,
            },
        )

        await db.commit()
    except Exception as e:
        logger.error(f"Failed to persist analysis results to database: {e}", exc_info=True)
        await db.rollback()

    return response
