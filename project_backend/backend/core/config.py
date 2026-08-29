"""
FastAPI Backend Configuration.

Handles environment configurations, default endpoints, device selection,
and filesystem directories for uploads and reports.
"""

import os
from pathlib import Path
from typing import List
import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Multimodal Retinal Analysis System API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Host & Port
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Device Configuration: auto, cuda, or cpu
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    UPLOAD_DIR: Path = BACKEND_DIR / "uploads"
    REPORTS_DIR: Path = BACKEND_DIR / "reports"


settings = Settings()

# Ensure working directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
