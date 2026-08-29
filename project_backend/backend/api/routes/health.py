"""
Health & Model Diagnostic Status Endpoints.
"""

from fastapi import APIRouter
from ...core.config import settings
from ...schemas.output_schema import HealthResponse, ModelStatusResponse
from ...services.model_service import ModelManager

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Verify backend API health status."""
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        device=settings.DEVICE,
    )


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """Inspect loading status of all deep learning pipeline modules."""
    manager = ModelManager.get_instance()
    status_dict = manager.get_model_status()
    return ModelStatusResponse(**status_dict)
