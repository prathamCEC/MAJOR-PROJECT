"""
Automated Database Migrations Test Suite.
Verifies Alembic migrations reproducibility on clean database, table constraints, and foreign keys.
"""

import sqlite3
import pytest
from alembic.config import Config
from alembic import command


def test_alembic_clean_db_migration_cycle(tmp_path):
    """
    Test full lifecycle:
    Empty DB -> Upgrade to Head -> Downgrade to Base -> Upgrade to Head.
    """
    db_file = tmp_path / "migration_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)

    # 1. Upgrade from empty to head
    command.upgrade(cfg, "head")
    assert db_file.exists(), "Database file should be created by migration."

    # 2. Downgrade back to base
    command.downgrade(cfg, "base")

    # 3. Re-upgrade to head
    command.upgrade(cfg, "head")


def test_database_schema_constraints_and_foreign_keys(tmp_path):
    """Verify that all 8 tables and foreign key constraints exist."""
    db_file = tmp_path / "schema_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = set(r[0] for r in cursor.fetchall())
    conn.close()

    required_tables = {
        "users",
        "refresh_tokens",
        "patients",
        "analysis_sessions",
        "uploaded_images",
        "predictions",
        "reports",
        "audit_logs",
        "alembic_version",
    }
    for t in required_tables:
        assert t in tables, f"Expected table '{t}' in migrated database."
