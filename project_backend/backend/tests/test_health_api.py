"""
Tests for FastAPI health and model diagnostic endpoints.
"""

import httpx
import pytest
from backend.main import app


@pytest.mark.anyio
async def test_health_endpoint():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "retinal" in data["service"].lower()
        assert "version" in data


@pytest.mark.anyio
async def test_model_status_endpoint():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/model-status")
        assert response.status_code == 200
        data = response.json()
        assert "phase4_octa" in data
        assert "phase8" in data
        assert "phase11" in data
        assert data["phase4_octa"] in ("loaded", "missing")
