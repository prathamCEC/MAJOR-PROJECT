"""
Relational Database Models for Retinal AI System.
Defines Users, Refresh Tokens, Patients, Analysis Sessions, Uploaded Images,
Predictions, Reports, and Security Audit Logs.
"""

from datetime import datetime, timezone
import enum
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, utc_now


class UserRole(str, enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class AnalysisStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ModalityEnum(str, enum.Enum):
    OCTA = "octa"
    OCTB = "octb"
    FUNDUS = "fundus"


class RiskTierEnum(str, enum.Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class User(Base, TimestampMixin):
    """Authenticated user / clinician in the system."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    patients: Mapped[List["Patient"]] = relationship("Patient", back_populates="owner", cascade="all, delete-orphan")
    analysis_sessions: Mapped[List["AnalysisSession"]] = relationship("AnalysisSession", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")
    reports: Mapped[List["Report"]] = relationship("Report", back_populates="user")


class RefreshToken(Base):
    """Cryptographically hashed refresh token for session lifecycle."""
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")


class Patient(Base, TimestampMixin):
    """Patient medical profile and clinical health record."""
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_code: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    owner_user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Demographics & Tabular Clinical Variables
    age_group: Mapped[str] = mapped_column(String(50), default="O_CD", nullable=False)
    gender: Mapped[int] = mapped_column(Integer, default=1, nullable=False)  # 1=Male, 0=Female
    education_years: Mapped[float] = mapped_column(Float, default=16.0, nullable=False)
    bmi: Mapped[float] = mapped_column(Float, default=26.5, nullable=False)
    obese: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    hypertension: Mapped[int] = mapped_column(Integer, default=1, nullable=False)  # 1=Positive, 0=Negative
    diabetes_type2: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 1=Positive, 0=Negative
    smoking_ever: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    smoking_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    alcohol_ever: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    alcohol_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="patients")
    analysis_sessions: Mapped[List["AnalysisSession"]] = relationship("AnalysisSession", back_populates="patient", cascade="all, delete-orphan")
    reports: Mapped[List["Report"]] = relationship("Report", back_populates="patient")


class AnalysisSession(Base, TimestampMixin):
    """Lifecycle record for an individual patient diagnostic evaluation."""
    __tablename__ = "analysis_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_uuid: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    status: Mapped[AnalysisStatus] = mapped_column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False, index=True)
    modalities_requested: Mapped[str] = mapped_column(String(100), nullable=False)  # Comma-separated (e.g. 'octa,fundus')
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analysis_sessions")
    patient: Mapped["Patient"] = relationship("Patient", back_populates="analysis_sessions")
    uploaded_images: Mapped[List["UploadedImage"]] = relationship("UploadedImage", back_populates="session", cascade="all, delete-orphan")
    prediction: Mapped[Optional["Prediction"]] = relationship("Prediction", back_populates="session", uselist=False, cascade="all, delete-orphan")
    report: Mapped[Optional["Report"]] = relationship("Report", back_populates="session", uselist=False, cascade="all, delete-orphan")


class UploadedImage(Base):
    """Metadata and quality assessment records for an uploaded retinal scan."""
    __tablename__ = "uploaded_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    modality: Mapped[ModalityEnum] = mapped_column(Enum(ModalityEnum), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_decision: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=func.now())

    # Relationships
    session: Mapped["AnalysisSession"] = relationship("AnalysisSession", back_populates="uploaded_images")


class Prediction(Base, TimestampMixin):
    """Deep learning multi-task prediction results, uncertainty, and XAI metadata."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), unique=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)

    # Stroke Assessment Metrics
    stroke_probability: Mapped[float] = mapped_column(Float, nullable=False)
    stroke_risk_tier: Mapped[RiskTierEnum] = mapped_column(Enum(RiskTierEnum), nullable=False)
    stroke_confidence_percent: Mapped[float] = mapped_column(Float, nullable=False)
    stroke_variance: Mapped[float] = mapped_column(Float, nullable=False)
    stroke_entropy: Mapped[float] = mapped_column(Float, nullable=False)

    # Alzheimer's Assessment Metrics
    alzheimer_probability: Mapped[float] = mapped_column(Float, nullable=False)
    alzheimer_risk_tier: Mapped[RiskTierEnum] = mapped_column(Enum(RiskTierEnum), nullable=False)
    alzheimer_confidence_percent: Mapped[float] = mapped_column(Float, nullable=False)
    alzheimer_variance: Mapped[float] = mapped_column(Float, nullable=False)
    alzheimer_entropy: Mapped[float] = mapped_column(Float, nullable=False)

    # Multi-Task Synthesis
    overall_risk_level: Mapped[RiskTierEnum] = mapped_column(Enum(RiskTierEnum), nullable=False)
    shap_summary_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gradcam_paths_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    session: Mapped["AnalysisSession"] = relationship("AnalysisSession", back_populates="prediction")


class Report(Base, TimestampMixin):
    """Generated clinical reports (JSON and PDF)."""
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), unique=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    pdf_path: Mapped[str] = mapped_column(String(512), nullable=False)
    json_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # Relationships
    session: Mapped["AnalysisSession"] = relationship("AnalysisSession", back_populates="report")
    user: Mapped["User"] = relationship("User", back_populates="reports")
    patient: Mapped["Patient"] = relationship("Patient", back_populates="reports")


class AuditLog(Base):
    """System security and clinical activity audit trail."""
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, server_default=func.now(), index=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
