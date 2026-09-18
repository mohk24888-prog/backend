from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import LoginRequest, RegisterRequest, Token, User
from db.models import User as UserModel
from db.session import get_session
from core.security.auth import create_access_token, hash_password, verify_password

router = APIRouter()


@router.post("/register", response_model=Token)
async def register(
    payload: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Token:
    existing = (await db.execute(select(UserModel).where(UserModel.email == payload.email))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = UserModel(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    await db.flush()
    role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
    token = create_access_token(subject=str(user.id), extra_claims={"role": role_value})
    return Token(access_token=token)


@router.post("/login", response_model=Token)
async def login(
    payload: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Token:
    user = (await db.execute(select(UserModel).where(UserModel.email == payload.email))).scalar_one_or_none()
    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
    token = create_access_token(subject=str(user.id), extra_claims={"role": role_value})
    return Token(access_token=token)


@router.get("/me", response_model=User)
async def me(current_user: Annotated[UserModel, Depends(get_current_user)]) -> UserModel:
    return current_user
