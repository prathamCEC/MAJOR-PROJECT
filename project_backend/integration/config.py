"""
Configuration for Phase 2 -> Phase 3 Integrated Workflow.

Defines dataset routing paths (raw, processed, approved, rejected) and execution defaults.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get project_backend absolute root path."""
    return Path(__file__).resolve().parent.parent


def get_raw_dir(modality: Optional[str] = None) -> Path:
    base = get_project_root() / "datasets" / "raw"
    return base / modality.lower() if modality else base


def get_processed_dir(modality: Optional[str] = None) -> Path:
    base = get_project_root() / "datasets" / "processed"
    return base / modality.lower() if modality else base


def get_approved_dir(modality: Optional[str] = None) -> Path:
    base = get_project_root() / "datasets" / "approved"
    return base / modality.lower() if modality else base


def get_rejected_dir(modality: Optional[str] = None) -> Path:
    base = get_project_root() / "datasets" / "rejected"
    return base / modality.lower() if modality else base


def get_integration_log_dir() -> Path:
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
