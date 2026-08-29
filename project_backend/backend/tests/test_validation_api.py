"""
Tests for FastAPI input validation and error responses.
"""

import httpx
import pytest
from backend.main import app


@pytest.mark.anyio
async def test_invalid_clinical_group_raises_422():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Send invalid group 'INVALID_GROUP'
        response = await client.post(
            "/api/v1/analyze",
            data={
                "patient_id": "TEST_INVALID",
                "Old_groups": "INVALID_GROUP",
                "Gender": 1,
                "Education": 16.0,
                "BMI": 25.0,
            },
        )
        assert response.status_code == 422
        assert "detail" in response.json()


@pytest.mark.anyio
async def test_invalid_education_raises_422():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Send out of range education -5.0
        response = await client.post(
            "/api/v1/analyze",
            data={
                "patient_id": "TEST_INVALID",
                "Old_groups": "O_CD",
                "Gender": 1,
                "Education": -5.0,
                "BMI": 25.0,
            },
        )
        assert response.status_code == 422
