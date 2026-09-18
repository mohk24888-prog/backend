from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import ProfileCreate, ProfileRead, ProfileUpdate, User
from db.models import Profile as ProfileModel, User as UserModel

router = APIRouter()


@router.get("/me", response_model=User)
async def get_me(current_user: Annotated[UserModel, Depends(get_current_user)]) -> UserModel:
    return current_user


@router.get("/me/profile", response_model=ProfileRead)
async def get_profile(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> ProfileModel:
    profile = (await db.execute(select(ProfileModel).where(ProfileModel.user_id == current_user.id))).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return profile


@router.post("/me/profile", response_model=ProfileRead)
async def create_profile(
    payload: ProfileCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> ProfileModel:
    existing = (await db.execute(select(ProfileModel).where(ProfileModel.user_id == current_user.id))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile already exists")

    profile = ProfileModel(
        user_id=current_user.id,
        full_name=payload.full_name,
        phone=payload.phone,
        bio=payload.bio,
    )
    db.add(profile)
    await db.flush()
    return profile


@router.patch("/me/profile", response_model=ProfileRead)
async def update_profile(
    payload: ProfileUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> ProfileModel:
    profile = (await db.execute(select(ProfileModel).where(ProfileModel.user_id == current_user.id))).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(profile, field, value)
    await db.flush()
    return profile
