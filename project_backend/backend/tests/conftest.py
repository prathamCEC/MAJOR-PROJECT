"""
Pytest configuration for backend async tests.
"""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
