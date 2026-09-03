"""
REST API Client for communicating between Streamlit and FastAPI Backend.
Handles authentication, patient management, analysis runs, report downloads, and diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class RetinalAIClient:
    """
    Client connecting frontend to FastAPI backend endpoints.
    """

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def _headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

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

    # -------------------------------------------------------------
    # Authentication Methods
    # -------------------------------------------------------------
    def login(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """Authenticate with POST /api/v1/auth/login."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/auth/login",
                json={"username_or_email": username_or_email, "password": password},
                timeout=10,
            )
            if r.status_code == 200:
                return {"status": "ok", "data": r.json()}
            elif r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "60")
                return {"status": "rate_limited", "detail": f"Too many login attempts. Please wait {retry_after} seconds."}
            else:
                detail = r.json().get("detail", "Invalid username or password.") if r.text else "Login failed."
                return {"status": "error", "detail": detail}
        except Exception as e:
            return {"status": "error", "detail": f"Connection error: {str(e)}"}

    def register(self, email: str, username: str, password: str, full_name: str) -> Dict[str, Any]:
        """Create new clinician user with POST /api/v1/auth/register."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/auth/register",
                json={"email": email, "username": username, "password": password, "full_name": full_name},
                timeout=10,
            )
            if r.status_code in (200, 201):
                return {"status": "ok", "data": r.json()}
            elif r.status_code == 429:
                return {"status": "rate_limited", "detail": "Too many registration attempts. Please wait a moment."}
            else:
                detail = r.json().get("detail", "Registration failed.") if r.text else "Registration error."
                return {"status": "error", "detail": detail}
        except Exception as e:
            return {"status": "error", "detail": f"Connection error: {str(e)}"}

    def logout(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Log out user with POST /api/v1/auth/logout."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/auth/logout",
                headers=self._headers(token),
                timeout=5,
            )
            return {"status": "ok"}
        except Exception:
            return {"status": "ok"}

    def get_me(self, token: str) -> Dict[str, Any]:
        """Fetch current user profile with GET /api/v1/auth/me."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/auth/me",
                headers=self._headers(token),
                timeout=5,
            )
            if r.status_code == 200:
                return {"status": "ok", "data": r.json()}
            elif r.status_code == 401:
                return {"status": "unauthorized", "detail": "Session expired."}
            else:
                return {"status": "error", "detail": "Failed to fetch profile."}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # -------------------------------------------------------------
    # Patient Management Methods
    # -------------------------------------------------------------
    def list_patients(self, token: str, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query GET /api/v1/patients."""
        try:
            params = {}
            if search:
                params["search"] = search
            r = requests.get(
                f"{self.base_url}/api/v1/patients/",
                headers=self._headers(token),
                params=params,
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return []
        except Exception:
            return []

    def create_patient(self, token: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new patient profile via POST /api/v1/patients/."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/patients/",
                headers=self._headers(token),
                json=patient_data,
                timeout=10,
            )
            if r.status_code in (200, 201):
                return {"status": "ok", "data": r.json()}
            detail = r.json().get("detail", "Failed to create patient.") if r.text else "Error"
            return {"status": "error", "detail": detail}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def get_patient(self, token: str, patient_code: str) -> Optional[Dict[str, Any]]:
        """Query GET /api/v1/patients/{patient_code}."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/patients/{patient_code}",
                headers=self._headers(token),
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            return None

    # -------------------------------------------------------------
    # Analysis & Inference Methods
    # -------------------------------------------------------------
    def run_analysis(
        self,
        clinical_data: Dict[str, Any],
        octa_file: Optional[Tuple[str, bytes, str]] = None,
        octb_file: Optional[Tuple[str, bytes, str]] = None,
        fundus_file: Optional[Tuple[str, bytes, str]] = None,
        token: Optional[str] = None,
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

        headers = self._headers(token)

        try:
            r = requests.post(
                f"{self.base_url}/api/v1/analyze",
                headers=headers,
                data=clinical_data,
                files=files if files else None,
                timeout=180,
            )
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 401:
                return {"status": "unauthorized", "detail": "Session expired. Please log in."}
            elif r.status_code == 422:
                err = r.json().get("detail", "Validation error")
                return {"status": "error", "detail": err}
            else:
                err_msg = r.text if "r" in locals() else "Analysis error"
                return {"status": "error", "detail": f"HTTP Error {r.status_code}: {err_msg}"}
        except requests.HTTPError as e:
            err_msg = r.text if "r" in locals() else str(e)
            return {"status": "error", "detail": f"HTTP Error {r.status_code}: {err_msg}"}
        except Exception as e:
            return {"status": "error", "detail": f"Connection error: {str(e)}"}

    def list_analyses(self, token: str, patient_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query GET /api/v1/analyses/."""
        try:
            params = {}
            if patient_code:
                params["patient_code"] = patient_code
            r = requests.get(
                f"{self.base_url}/api/v1/analyses/",
                headers=self._headers(token),
                params=params,
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return []
        except Exception:
            return []

    # -------------------------------------------------------------
    # Report Methods
    # -------------------------------------------------------------
    def list_reports(self, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query GET /api/v1/reports."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/reports",
                headers=self._headers(token),
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return []
        except Exception:
            return []

    def download_pdf_report(self, report_id: str, token: Optional[str] = None) -> Optional[bytes]:
        """
        Retrieve generated report PDF bytes via GET /api/v1/report/{report_id}/pdf.
        """
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/report/{report_id}/pdf",
                headers=self._headers(token),
                timeout=15,
            )
            r.raise_for_status()
            return r.content
        except Exception:
            return None

    def download_json_report(self, report_id: str, token: Optional[str] = None) -> Optional[bytes]:
        """
        Retrieve generated report JSON bytes via GET /api/v1/report/{report_id}/json.
        """
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/report/{report_id}/json",
                headers=self._headers(token),
                timeout=15,
            )
            r.raise_for_status()
            return r.content
        except Exception:
            return None
