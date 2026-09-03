"""
Pydantic Schemas for User Authentication and Account Lifecycle.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from ..db.models import UserRole


class UserRegisterRequest(BaseModel):
    """Payload for user registration."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)


class UserLoginRequest(BaseModel):
    """Payload for user login."""
    username_or_email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Authentication token response with access and refresh tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Payload to refresh an expired access token."""
    refresh_token: str


class UserProfileResponse(BaseModel):
    """Public user profile data."""
    id: int
    email: EmailStr
    username: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Standard message response."""
    message: str
    detail: Optional[str] = None
