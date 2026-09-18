from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import PlayerRead
from db.models import Player as PlayerModel, User as UserModel

router = APIRouter()


@router.get("/players", response_model=List[PlayerRead])
async def discover_players(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    q: str | None = Query(default=None),
    position: str | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[PlayerModel]:
    query = select(PlayerModel).where(PlayerModel.is_public.is_(True))
    if q:
        query = query.where(
            or_(
                PlayerModel.first_name.ilike(f"%{q}%"),
                PlayerModel.last_name.ilike(f"%{q}%"),
                PlayerModel.academy.ilike(f"%{q}%"),
            )
        )
    if position:
        query = query.where(PlayerModel.position == position)

    result = await db.execute(query.limit(limit).offset(offset))
    return result.scalars().all()
