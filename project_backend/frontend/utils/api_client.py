"""
REST API Client for communicating between Streamlit and FastAPI Backend.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class RetinalAIClient:
    """
    Client connecting frontend to FastAPI backend endpoints.
    """

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Query GET /health."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_model_status(self) -> Dict[str, Any]:
        """Query GET /model-status."""
        try:
            r = requests.get(f"{self.base_url}/model-status", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_analysis(
        self,
        clinical_data: Dict[str, Any],
        octa_file: Optional[Tuple[str, bytes, str]] = None,
        octb_file: Optional[Tuple[str, bytes, str]] = None,
        fundus_file: Optional[Tuple[str, bytes, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send multipart POST /api/v1/analyze request.
        """
        files = {}
        if octa_file:
            files["octa_file"] = octa_file
        if octb_file:
            files["octb_file"] = octb_file
        if fundus_file:
            files["fundus_file"] = fundus_file

        try:
            r = requests.post(
                f"{self.base_url}/api/v1/analyze",
                data=clinical_data,
                files=files if files else None,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            err_msg = r.text if "r" in locals() else str(e)
            return {"status": "error", "detail": f"HTTP Error {r.status_code}: {err_msg}"}
        except Exception as e:
            return {"status": "error", "detail": f"Connection error: {str(e)}"}

    def download_pdf_report(self, report_id: str) -> Optional[bytes]:
        """
        Retrieve generated report PDF bytes via GET /api/v1/report/{report_id}/pdf.
        """
        try:
            r = requests.get(f"{self.base_url}/api/v1/report/{report_id}/pdf", timeout=15)
            r.raise_for_status()
            return r.content
        except Exception as e:
            return None
