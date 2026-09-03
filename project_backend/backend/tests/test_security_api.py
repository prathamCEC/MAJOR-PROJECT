"""
Comprehensive Security and Vulnerability Automated Test Suite.
Verifies Rate Limiting, SQL Injection Resistance, XSS Handling, Auth Boundaries,
Path Traversal Defense, and File Upload Validation.
"""

from io import BytesIO
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from backend.core.config import settings
from backend.core.rate_limit import rate_limiter_store
from backend.core.security import create_access_token
from backend.main import app


@pytest.fixture(autouse=True)
def reset_rate_limit_state():
    """Clear rate limiter timestamps before each test."""
    with rate_limiter_store._lock:
        rate_limiter_store._requests.clear()
    yield


@pytest.mark.anyio
async def test_rate_limiting_on_login():
    """Verify that exceeding login threshold returns HTTP 429 Too Many Requests."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Max limit is 5 requests per 60s
        for _ in range(5):
            res = await ac.post(
                "/api/v1/auth/login",
                json={"username_or_email": "wrong_user@test.org", "password": "WrongPassword123!"},
            )
            assert res.status_code in (401, 400)

        # 6th attempt must be rejected with 429
        blocked_res = await ac.post(
            "/api/v1/auth/login",
            json={"username_or_email": "wrong_user@test.org", "password": "WrongPassword123!"},
        )
        assert blocked_res.status_code == 429
        assert "Retry-After" in blocked_res.headers
        assert "Too many requests" in blocked_res.json()["detail"]


@pytest.mark.anyio
async def test_sql_injection_resilience():
    """Verify SQL injection payloads in search and patient codes are safely parameterized."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        token = create_access_token(subject=1, role="USER")
        headers = {"Authorization": f"Bearer {token}"}

        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE patients; --",
            "1 UNION SELECT null, null, null --",
        ]

        for payload in sql_payloads:
            # Search endpoint
            res = await ac.get(f"/api/v1/patients/?search={payload}", headers=headers)
            assert res.status_code == 200
            # Result should simply be empty list, database intact
            assert isinstance(res.json(), list)

            # Patient detail endpoint
            detail_res = await ac.get(f"/api/v1/patients/{payload}", headers=headers)
            assert detail_res.status_code == 404


@pytest.mark.anyio
async def test_xss_payload_safety():
    """Verify XSS strings in patient name and clinical fields are treated as literal text."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        token = create_access_token(subject=1, role="USER")
        headers = {"Authorization": f"Bearer {token}"}

        xss_code = "PAT_XSS_01"
        xss_name = "<script>alert('xss')</script>"

        # Create
        res = await ac.post(
            "/api/v1/patients/",
            headers=headers,
            json={
                "patient_code": xss_code,
                "full_name": xss_name,
                "age_group": "O_CD",
                "gender": 1,
                "education_years": 16.0,
                "bmi": 26.5,
                "obese": 0.0,
                "hypertension": 1,
                "diabetes_type2": 0,
                "smoking_ever": 1,
                "smoking_current": 0,
                "alcohol_ever": 1,
                "alcohol_current": 0,
            },
        )
        assert res.status_code in (201, 400)
        if res.status_code == 201:
            data = res.json()
            assert data["full_name"] == xss_name

        # Clean up
        await ac.delete(f"/api/v1/patients/{xss_code}", headers=headers)


@pytest.mark.anyio
async def test_invalid_and_forged_tokens():
    """Verify forged or malformed tokens return HTTP 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        forged_headers = {"Authorization": "Bearer forged.jwt.signature"}
        res = await ac.get("/api/v1/auth/me", headers=forged_headers)
        assert res.status_code == 401

        empty_headers = {"Authorization": "Bearer "}
        res = await ac.get("/api/v1/auth/me", headers=empty_headers)
        assert res.status_code == 401


@pytest.mark.anyio
async def test_path_traversal_defense():
    """Verify path traversal characters in report downloads are rejected."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        token = create_access_token(subject=1, role="USER")
        headers = {"Authorization": f"Bearer {token}"}

        traversal_attempts = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "....//....//report",
        ]
        for path in traversal_attempts:
            res = await ac.get(f"/api/v1/report/{path}/pdf", headers=headers)
            assert res.status_code in (404, 422)


@pytest.mark.anyio
async def test_invalid_image_upload():
    """Verify corrupted or non-image files are rejected with HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        corrupted_bytes = b"NOT_A_REAL_IMAGE_CORRUPT_BYTES_XYZ"
        files = {"octa_file": ("test.png", corrupted_bytes, "image/png")}
        data = {"patient_id": "TEST_PAT_CORRUPT"}

        res = await ac.post("/api/v1/analyze", data=data, files=files)
        assert res.status_code == 400
        assert "Security signature verification failed" in res.json()["detail"]


@pytest.mark.anyio
async def test_oversized_image_upload(monkeypatch):
    """Verify uploads exceeding MAX_UPLOAD_SIZE_BYTES return HTTP 413."""
    # Temporarily set limit to 100 bytes to test rejection
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 100)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        large_bytes = b"A" * 500
        files = {"octa_file": ("test.png", large_bytes, "image/png")}
        data = {"patient_id": "TEST_PAT_LARGE"}

        res = await ac.post("/api/v1/analyze", data=data, files=files)
        assert res.status_code == 413
        assert "exceeds maximum allowed size" in res.json()["detail"]
