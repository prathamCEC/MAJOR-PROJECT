"""
User Authentication Endpoints: Register, Login, Refresh, Me, Logout.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import settings
from ...core.rate_limit import rate_limit
from ...core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_password_hash,
    hash_token,
    validate_password_strength,
    verify_password,
)
from ...db.models import RefreshToken, User, UserRole
from ...db.session import get_db
from ...schemas.auth_schema import (
    MessageResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)
from ..deps import get_current_user, record_audit_log

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(max_requests=5, window_seconds=60))],
)
async def register(
    req: UserRegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new clinician/user account.
    Enforces password complexity and unique username/email.
    """
    # 1. Validate password strength
    is_valid, err_msg = validate_password_strength(req.password)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    # 2. Check duplicate email or username
    existing = await db.execute(
        select(User).where((User.email == req.email.lower()) | (User.username == req.username.lower()))
    )
    if existing.scalar_one_or_none():
        # Generic error message to prevent account enumeration
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email or username already exists.",
        )

    # 3. Create user (always Role.USER through public registration)
    new_user = User(
        email=req.email.lower(),
        username=req.username.lower(),
        hashed_password=get_password_hash(req.password),
        full_name=req.full_name,
        role=UserRole.USER,
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # 4. Audit Log
    client_ip = request.client.host if request.client else None
    await record_audit_log(
        db=db,
        action="REGISTER",
        user_id=new_user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
        details={"username": new_user.username, "email": new_user.email},
    )

    return new_user


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(rate_limit(max_requests=5, window_seconds=60))],
)
async def login(
    req: UserLoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user with username/email and password.
    Returns access and refresh tokens and sets secure HTTP-only cookie.
    """
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # 1. Find user by email or username
    ident = req.username_or_email.strip().lower()
    result = await db.execute(
        select(User).where((User.email == ident) | (User.username == ident))
    )
    user = result.scalar_one_or_none()

    # 2. Verify password (generic failure to prevent user enumeration)
    if not user or not verify_password(req.password, user.hashed_password) or not user.is_active:
        await record_audit_log(
            db=db,
            action="LOGIN_FAILED",
            user_id=user.id if user else None,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"identifier_attempted": ident},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Please check your username/email and password.",
        )

    # 3. Generate tokens
    access_token = create_access_token(subject=user.id, role=user.role.value)
    refresh_token, refresh_exp = create_refresh_token(subject=user.id)

    # 4. Persist refresh token hash
    token_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        expires_at=refresh_exp,
        is_revoked=False,
    )
    db.add(token_record)
    await db.commit()

    # 5. Set secure HTTP-only cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False if settings.ENVIRONMENT == "development" else True,
    )

    # 6. Audit Log
    await record_audit_log(
        db=db,
        action="LOGIN_SUCCESS",
        user_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    req: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a new access token using a valid, unrevoked refresh token."""
    payload = decode_refresh_token(req.refresh_token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")

    user_id = int(payload.get("sub"))
    t_hash = hash_token(req.refresh_token)

    # Verify token exists and is not revoked
    res = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == t_hash,
            RefreshToken.user_id == user_id,
            RefreshToken.is_revoked == False,
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
    )
    token_record = res.scalar_one_or_none()
    if not token_record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked or expired.")

    # Get user
    user_res = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user = user_res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    new_access_token = create_access_token(subject=user.id, role=user.role.value)
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=req.refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Retrieve profile of the currently authenticated user."""
    return current_user


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Logout user and clear authentication cookies."""
    response.delete_cookie(key="access_token")
    client_ip = request.client.host if request.client else None

    await record_audit_log(
        db=db,
        action="LOGOUT",
        user_id=current_user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
    )

    return MessageResponse(message="Successfully logged out.")
