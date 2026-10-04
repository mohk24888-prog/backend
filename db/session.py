from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import Base

logger = logging.getLogger(__name__)

# Primary engine backed by the configured DATABASE_URL.
engine: Any = None
async_session_factory: Any = None
_fallback_engine: Any = None
_fallback_factory: Any = None
_active_url: Optional[str] = None


def _is_postgres(url: str) -> bool:
    try:
        return urlparse(url).scheme.startswith("postgresql")
    except Exception:
        return False


def _sqlite_url() -> str:
    path = os.path.join("/tmp", "footiq_fallback.db")
    return f"sqlite+aiosqlite:///{path}"


async def _probe(url: str) -> bool:
    try:
        tmp = create_async_engine(url, echo=False, future=True)
        async with tmp.begin() as conn:
            await conn.execute(text("SELECT 1"))
        await tmp.dispose()
        return True
    except Exception as exc:
        logger.debug("DB probe failed for %s: %s", url, type(exc).__name__)
        return False


async def init_db() -> None:
    global engine, async_session_factory, _fallback_engine, _fallback_factory, _active_url

    primary = str(settings.database_url)
    _active_url = primary

    if await _probe(primary):
        engine = create_async_engine(primary, echo=False, future=True)
        async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        logger.info("Connected to primary database: %s", primary.split("@")[-1])
    else:
        logger.warning("Primary database unreachable, falling back to SQLite")
        fallback = _sqlite_url()
        _fallback_engine = create_async_engine(fallback, echo=False, future=True)
        _fallback_factory = async_sessionmaker(
            bind=_fallback_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        engine = _fallback_engine
        async_session_factory = _fallback_factory
        _active_url = fallback

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def drop_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
