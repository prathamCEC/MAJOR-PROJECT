"""
Tests for FastAPI analyze and PDF report download endpoints.
"""

from io import BytesIO
from pathlib import Path
import httpx
import numpy as np
from PIL import Image
import pytest

from backend.main import app


@pytest.mark.anyio
async def test_analyze_endpoint_and_report_download():
    # 1. Create a dummy PNG image buffer
    img = Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8))
    img_buf = BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    # 2. Call POST /api/v1/analyze
    form_data = {
        "patient_id": "TEST_API_PATIENT",
        "Old_groups": "O_CD",
        "Gender": "1",
        "Education": "16.0",
        "BMI": "26.5",
        "Obese": "0.0",
        "EtOH_ever": "1",
        "EtOH_current": "0",
        "Smoking_ever": "1",
        "Smoking_current": "0",
        "HTN": "1",
        "DM2": "0",
    }
    files = {
        "octa_file": ("test_octa.png", img_bytes, "image/png"),
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        response = await client.post("/api/v1/analyze", data=form_data, files=files)
        assert response.status_code == 200, f"Error: {response.text}"
        data = response.json()

        assert data["status"] == "success"
        assert data["patient_id"] == "TEST_API_PATIENT"
        assert "stroke_prediction" in data and "alzheimer_prediction" in data
        assert "pdf_download_url" in data

        report_id = data["report_id"]

        # 3. Test GET /api/v1/report/{report_id}/pdf
        pdf_resp = await client.get(f"/api/v1/report/{report_id}/pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert len(pdf_resp.content) > 1000

        # 4. Test GET /api/v1/report/{report_id}/json
        json_resp = await client.get(f"/api/v1/report/{report_id}/json")
        assert json_resp.status_code == 200
        assert json_resp.headers["content-type"] == "application/json"
