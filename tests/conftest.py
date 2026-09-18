from __future__ import annotations

import os
os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.models import Base
from db.session import get_session


@pytest.fixture(scope="module")
async def engine():
    _engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield _engine
    await _engine.dispose()


@pytest.fixture(scope="module")
async def session(engine):
    _factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with _factory() as session:
        yield session
        await session.rollback()
