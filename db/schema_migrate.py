from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings

logger = logging.getLogger(__name__)

# Columns that were added to the model after the initial deployment and may
# therefore be missing from an existing database. Each entry is
# (table_name, column_name, column_definition_sql).
_COLUMNS_TO_RECONCILE = [
    (
        "players",
        "verification_status",
        "VARCHAR(50) NOT NULL DEFAULT 'unverified'",
    ),
    (
        "players",
        "is_public",
        "BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    (
        "video_uploads",
        "updated_at",
        "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
    ),
]


async def reconcile_columns(conn: Any) -> None:
    """Add any model columns that are missing from the live database.

    create_all() only creates tables; it never adds columns to an existing
    table. When we add a column to the model, the deployed Supabase database
    therefore drifts. This lightweight migration reconciles the drift without
    Alembic or a migration framework.
    """
    url = urlparse(str(settings.database_url))
    is_postgres = url.scheme.startswith("postgresql")

    if not is_postgres:
        # SQLite / other: try-and-ignore is good enough for tests.
        for table, column, _ in _COLUMNS_TO_RECONCILE:
            try:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column}"))
            except Exception:
                pass
        return

    # Postgres: use information_schema so we only add genuinely missing
    # columns and don't trip over the IF NOT EXISTS syntax that varies by
    # Postgres version.
    rows = (await conn.execute(
        text(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name IN (:t0, :t1) "
            "AND column_name IN (:c0, :c1, :c2)"
        ),
        {
            "t0": "players",
            "t1": "video_uploads",
            "c0": "verification_status",
            "c1": "is_public",
            "c2": "updated_at",
        },
    )).fetchall()

    present = {(r[0], r[1]) for r in rows}

    for table, column, definition in _COLUMNS_TO_RECONCILE:
        if (table, column) in present:
            continue
        try:
            await conn.execute(
                text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            )
            logger.info("Added column %s.%s", table, column)
        except Exception as exc:
            logger.warning("Could not add column %s.%s: %s", table, column, exc)
