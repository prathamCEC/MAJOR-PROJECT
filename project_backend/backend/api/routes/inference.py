"""
Patient Diagnostic Inference Endpoints.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ...core.logging_config import logger
from ...schemas.input_schema import PatientClinicalInput
from ...schemas.output_schema import AnalysisResponse
from ...services.inference_service import InferenceService

router = APIRouter(prefix="/api/v1", tags=["Multimodal Analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient_endpoint(
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
):
    """
    Execute full multimodal inference (Phase 2 to Phase 11) for a patient.
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

    # 2. Read Image Bytes
    octa_bytes = await octa_file.read() if octa_file else None
    octb_bytes = await octb_file.read() if octb_file else None
    fundus_bytes = await fundus_file.read() if fundus_file else None

    # 3. Execute Inference Workflow
    try:
        service = InferenceService()
        response = service.analyze_patient(
            clinical_input=clinical_input,
            octa_bytes=octa_bytes,
            octb_bytes=octb_bytes,
            fundus_bytes=fundus_bytes,
        )
        return response
    except Exception as e:
        logger.error(f"Inference execution failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")
