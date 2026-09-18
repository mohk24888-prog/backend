from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import PlayerCreate, PlayerRead, PlayerUpdate
from db.models import Player as PlayerModel, User as UserModel

router = APIRouter()


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid id")


@router.post("", response_model=PlayerRead)
async def create_player(
    payload: PlayerCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PlayerModel:
    player = PlayerModel(**payload.model_dump())
    db.add(player)
    await db.flush()
    return player


@router.get("/{player_id}", response_model=PlayerRead)
async def get_player(
    player_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PlayerModel:
    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == _uuid(player_id)))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")
    return player


@router.patch("/{player_id}", response_model=PlayerRead)
async def update_player(
    player_id: str,
    payload: PlayerUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PlayerModel:
    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == _uuid(player_id)))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(player, field, value)
    await db.flush()
    return player


@router.get("", response_model=list[PlayerRead])
async def list_players(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    q: str | None = Query(default=None),
    position: str | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[PlayerModel]:
    query = select(PlayerModel)
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
