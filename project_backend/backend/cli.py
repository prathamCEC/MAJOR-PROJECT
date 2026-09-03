"""
Command-Line Interface (CLI) for Retinal AI System Administration.
Provides secure commands to provision administrators and inspect database health.
"""

import argparse
import asyncio
import getpass
import sys
from sqlalchemy import select

from .core.security import get_password_hash, validate_password_strength
from .db.models import User, UserRole
from .db.session import AsyncSessionLocal
from .db.init_db import init_db


async def async_create_admin(email: str, username: str, password: str, full_name: str) -> None:
    """Safely create an administrator account with password validation."""
    await init_db()

    is_valid, err_msg = validate_password_strength(password)
    if not is_valid:
        print(f"\n[ERROR] Password does not meet security requirements:\n  {err_msg}")
        sys.exit(1)

    async with AsyncSessionLocal() as session:
        email_clean = email.strip().lower()
        user_clean = username.strip().lower()

        existing = await session.execute(
            select(User).where((User.email == email_clean) | (User.username == user_clean))
        )
        if existing.scalar_one_or_none():
            print(f"\n[ERROR] An account with email '{email_clean}' or username '{user_clean}' already exists.")
            sys.exit(1)

        admin = User(
            email=email_clean,
            username=user_clean,
            hashed_password=get_password_hash(password),
            full_name=full_name.strip(),
            role=UserRole.ADMIN,
            is_active=True,
        )
        session.add(admin)
        await session.commit()
        print(f"\n[SUCCESS] Administrator account '{user_clean}' ({email_clean}) provisioned successfully.")


def create_admin_cmd(args: argparse.Namespace) -> None:
    email = args.email
    username = args.username
    full_name = args.name or "System Administrator"
    password = args.password

    if not email:
        email = input("Enter Admin Email: ").strip()
    if not username:
        username = input("Enter Admin Username: ").strip()
    if not password:
        password = getpass.getpass("Enter Secure Admin Password: ")
        confirm = getpass.getpass("Confirm Admin Password: ")
        if password != confirm:
            print("\n[ERROR] Passwords do not match.")
            sys.exit(1)

    asyncio.run(async_create_admin(email, username, password, full_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retinal AI System CLI Management Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-admin sub-command
    admin_parser = subparsers.add_parser("create-admin", help="Safely provision an administrator account")
    admin_parser.add_argument("--email", "-e", type=str, help="Administrator email address")
    admin_parser.add_argument("--username", "-u", type=str, help="Administrator username")
    admin_parser.add_argument("--password", "-p", type=str, help="Administrator password")
    admin_parser.add_argument("--name", "-n", type=str, default="System Administrator", help="Administrator full name")

    parsed = parser.parse_args()
    if parsed.command == "create-admin":
        create_admin_cmd(parsed)


if __name__ == "__main__":
    main()
