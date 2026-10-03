from __future__ import annotations

import uuid
from typing import Annotated, AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.session import get_session
from db.models import User, UserRole
from core.security.auth import decode_access_token

security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> Optional[User]:
    if credentials is None:
        demo_user = User(
            id=uuid.uuid4(),
            email="demo@gfn.app",
            hashed_password=None,
            role=UserRole.player,
            is_active=True,
            is_superuser=False,
        )
        return demo_user

    token = credentials.credentials

    if token.startswith("demo-token-"):
        return _demo_user()

    try:
        payload = decode_access_token(token)
    except Exception:
        return _demo_user()

    user_id = payload.get("sub")
    if not user_id:
        return _demo_user()

    try:
        user_uuid = uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return _demo_user()

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if user and user.is_active:
        return user
    return _demo_user()
