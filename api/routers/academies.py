from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import AcademyRead
from db.models import Academy as AcademyModel, User as UserModel

router = APIRouter()


@router.get("/{academy_id}", response_model=AcademyRead)
async def get_academy(
    academy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AcademyModel:
    try:
        academy_uuid = uuid.UUID(str(academy_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid academy id")

    academy = (await db.execute(select(AcademyModel).where(AcademyModel.id == academy_uuid))).scalar_one_or_none()
    if not academy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Academy not found")
    return academy
