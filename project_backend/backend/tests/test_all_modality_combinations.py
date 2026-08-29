"""
Comprehensive Test for All 7 Retinal Modality Combinations + Zero-Modality Rejection.
"""

from io import BytesIO
import httpx
import numpy as np
from PIL import Image
import pytest

from backend.main import app


def create_dummy_png_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.anyio
async def test_zero_modality_rejected():
    """Verify that submitting zero retinal scans is rejected with HTTP 422."""
    form_data = {
        "patient_id": "TEST_ZERO_MOD",
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
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/analyze", data=form_data)
        assert response.status_code == 422
        assert "at least one retinal imaging modality" in response.json()["detail"]


@pytest.mark.anyio
@pytest.mark.parametrize("modality_subset", [
    ["octa"],
    ["octb"],
    ["fundus"],
    ["octa", "octb"],
    ["octa", "fundus"],
    ["octb", "fundus"],
    ["octa", "octb", "fundus"],
])
async def test_all_7_modality_combinations(modality_subset):
    """
    Verify all 7 valid subsets of retinal modalities execute end-to-end (Phase 2 -> 11).
    """
    img_bytes = create_dummy_png_bytes()
    form_data = {
        "patient_id": f"PATIENT_{'_'.join(modality_subset).upper()}",
        "Old_groups": "O_CD",
        "Gender": "1",
        "Education": "16.0",
        "BMI": "27.0",
        "Obese": "0.0",
        "EtOH_ever": "1",
        "EtOH_current": "0",
        "Smoking_ever": "1",
        "Smoking_current": "0",
        "HTN": "1",
        "DM2": "0",
    }
    files = {}
    for m in modality_subset:
        files[f"{m}_file"] = (f"scan_{m}.png", img_bytes, "image/png")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=90.0) as client:
        response = await client.post("/api/v1/analyze", data=form_data, files=files)
        assert response.status_code == 200, f"Failed on {modality_subset}: {response.text}"
        data = response.json()

        assert data["status"] == "success"
        assert set(data["modalities_processed"]) == set(modality_subset)
        assert 0.0 <= data["stroke_prediction"]["probability"] <= 1.0
        assert 0.0 <= data["alzheimer_prediction"]["probability"] <= 1.0
        assert 0.0 <= data["stroke_uncertainty"]["confidence_percent"] <= 100.0

        # Verify PDF report retrieval
        rep_id = data["report_id"]
        pdf_resp = await client.get(f"/api/v1/report/{rep_id}/pdf")
        assert pdf_resp.status_code == 200
        assert len(pdf_resp.content) > 1000
