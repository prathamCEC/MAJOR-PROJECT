"""
FastAPI Backend Configuration.

Handles environment configurations, default endpoints, device selection,
security secrets, database connectivity, and filesystem directories.
"""

from pathlib import Path
from typing import List, Optional, Union
import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Multimodal Retinal Analysis System API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
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
    
    # Security & JWT Configurations
    JWT_SECRET_KEY: str = "dev-secret-key-please-override-in-production-env-2026-multimodal-retina"
    JWT_REFRESH_SECRET_KEY: str = "dev-refresh-secret-key-please-override-in-production-2026-retina"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Relational Database Connection (Default: Async SQLite, easily overridden with PostgreSQL)
    DATABASE_URL: str = "sqlite+aiosqlite:///./retinal_ai.db"
    
    # Initial Admin Account Settings (Optional - if set via env, will bootstrap safely; otherwise use CLI)
    FIRST_ADMIN_EMAIL: Optional[str] = None
    FIRST_ADMIN_USERNAME: Optional[str] = None
    FIRST_ADMIN_PASSWORD: Optional[str] = None
    FIRST_ADMIN_FULL_NAME: Optional[str] = "Lead Clinician Administrator"
    
    # Upload Security
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    ALLOWED_MIME_TYPES: List[str] = ["image/png", "image/jpeg", "image/tiff"]
    
    # Device Configuration: auto, cuda, or cpu
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    UPLOAD_DIR: Path = BACKEND_DIR / "uploads"
    REPORTS_DIR: Path = BACKEND_DIR / "reports"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure working directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
