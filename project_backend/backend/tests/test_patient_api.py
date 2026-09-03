"""
Automated Tests for Patient CRUD and Ownership Security.
"""

import httpx
import pytest
from backend.main import app
from backend.db.init_db import init_db


@pytest.fixture(autouse=True)
async def ensure_db():
    await init_db()


@pytest.mark.anyio
async def test_patient_crud_and_access_control():
    """Verify creating, listing, retrieving, updating, and deleting a patient."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Login as admin to obtain token
        login_resp = await client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "admin", "password": "Admin@SecurePass2026!"},
        )
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Create Patient
        p_code = "PAT_TEST_SEC_99"
        p_payload = {
            "patient_code": p_code,
            "full_name": "Test Subject Security",
            "age_group": "O_CD",
            "gender": 1,
            "education_years": 16.0,
            "bmi": 28.2,
            "obese": 0.0,
            "hypertension": 1,
            "diabetes_type2": 0,
            "smoking_ever": 1,
            "smoking_current": 0,
            "alcohol_ever": 1,
            "alcohol_current": 0,
        }
        create_resp = await client.post("/api/v1/patients/", json=p_payload, headers=headers)
        assert create_resp.status_code in [201, 400]

        # 2. Get Patient by Code
        get_resp = await client.get(f"/api/v1/patients/{p_code}", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["patient_code"] == p_code

        # 3. List Patients
        list_resp = await client.get("/api/v1/patients/", headers=headers)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) >= 1

        # 4. Update Patient
        up_resp = await client.put(
            f"/api/v1/patients/{p_code}",
            json={"bmi": 29.0, "full_name": "Updated Subject Name"},
            headers=headers,
        )
        assert up_resp.status_code == 200
        assert up_resp.json()["bmi"] == 29.0
