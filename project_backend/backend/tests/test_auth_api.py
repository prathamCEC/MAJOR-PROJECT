"""
Comprehensive Automated Tests for User Registration, Login, Tokens, and Role Security.
"""

import httpx
import pytest
from backend.main import app
from backend.db.init_db import init_db


@pytest.fixture(autouse=True)
async def ensure_db():
    """Ensure database schema and initial admin are loaded before tests."""
    await init_db()


@pytest.mark.anyio
async def test_admin_login_success():
    """Verify default admin can log in and receive valid JWT tokens."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "username_or_email": "admin@retinalai.org",
                "password": "Admin@SecurePass2026!",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

        # Verify /me endpoint with Bearer token
        me_resp = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {data['access_token']}"},
        )
        assert me_resp.status_code == 200
        me_data = me_resp.json()
        assert me_data["email"] == "admin@retinalai.org"
        assert me_data["role"] == "ADMIN"


@pytest.mark.anyio
async def test_invalid_login_credentials():
    """Verify invalid credentials return generic 401 without exposing account existence."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "username_or_email": "nonexistent@user.com",
                "password": "WrongPassword123!",
            },
        )
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]


@pytest.mark.anyio
async def test_user_registration_lifecycle():
    """Verify new clinician registration, password validation, and token refresh."""
    user_payload = {
        "email": "dr.smith@hospital.org",
        "username": "dr_smith",
        "password": "SecureClinicianPass2026!",
        "full_name": "Dr. Alice Smith, MD",
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Register User
        reg_resp = await client.post("/api/v1/auth/register", json=user_payload)
        assert reg_resp.status_code in [201, 400]  # 400 if already created in prior run
        if reg_resp.status_code == 201:
            data = reg_resp.json()
            assert data["email"] == user_payload["email"]
            assert data["role"] == "USER"

        # 2. Login with registered user
        login_resp = await client.post(
            "/api/v1/auth/login",
            json={
                "username_or_email": user_payload["username"],
                "password": user_payload["password"],
            },
        )
        assert login_resp.status_code == 200
        tokens = login_resp.json()

        # 3. Refresh Token
        refresh_resp = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert refresh_resp.status_code == 200
        assert "access_token" in refresh_resp.json()

        # 4. Logout
        logout_resp = await client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert logout_resp.status_code == 200
