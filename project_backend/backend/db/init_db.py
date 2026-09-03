"""
Database Initialization and Schema Bootstrapping.
Creates all relational database tables and safely bootstraps initial administrator account if configured.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .base import Base
from .session import engine, AsyncSessionLocal
from .models import User, UserRole
from ..core.config import settings
from ..core.logging_config import logger
from ..core.security import get_password_hash, validate_password_strength


async def init_db() -> None:
    """Create all relational tables and bootstrap initial admin user safely."""
    logger.info("Initializing relational database schema...")
    
    # 1. Create Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified / created successfully.")

    # 2. Bootstrap Initial Admin User if environment variables are provided
    async with AsyncSessionLocal() as session:
        # Check if an admin already exists in the system
        admin_query = await session.execute(select(User).where(User.role == UserRole.ADMIN).limit(1))
        existing_admin = admin_query.scalar_one_or_none()

        if not existing_admin:
            if settings.FIRST_ADMIN_EMAIL and settings.FIRST_ADMIN_PASSWORD:
                # Enforce password complexity
                is_valid, err_msg = validate_password_strength(settings.FIRST_ADMIN_PASSWORD)
                if not is_valid:
                    logger.error(f"Insecure FIRST_ADMIN_PASSWORD provided: {err_msg}. Admin creation aborted.")
                    return

                username = settings.FIRST_ADMIN_USERNAME or settings.FIRST_ADMIN_EMAIL.split("@")[0]
                full_name = settings.FIRST_ADMIN_FULL_NAME or "System Administrator"

                logger.info(f"Provisioning initial administrator account from environment: {settings.FIRST_ADMIN_EMAIL}")
                new_admin = User(
                    email=settings.FIRST_ADMIN_EMAIL.lower().strip(),
                    username=username.lower().strip(),
                    hashed_password=get_password_hash(settings.FIRST_ADMIN_PASSWORD),
                    full_name=full_name,
                    role=UserRole.ADMIN,
                    is_active=True,
                )
                session.add(new_admin)
                await session.commit()
                logger.info("Initial administrator account bootstrapped successfully.")
            else:
                logger.info(
                    "Notice: No administrator account currently provisioned. "
                    "You may provision an administrator using: python -m backend.cli create-admin"
                )
        else:
            logger.info("Verified active administrator account exists in database.")
