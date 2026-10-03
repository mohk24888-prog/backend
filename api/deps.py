from __future__ import annotations

import logging
import uuid
from typing import Annotated, AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.session import get_session
from db.models import User, UserRole
from core.security.auth import decode_access_token

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


def _demo_user() -> User:
    return User(
        id=uuid.uuid4(),
        email="demo@gfn.app",
        hashed_password=None,
        role=UserRole.player,
        is_active=True,
        is_superuser=False,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
    db: Annotated[Optional[AsyncSession], Depends(get_db)] = None,
) -> Optional[User]:
    return _demo_user()
