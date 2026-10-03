from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import defer
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


_DEMO_PLAYERS = [
    PlayerModel(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, "player:dz1"),
        first_name="Riyad",
        last_name="Mahrez",
        position="Right Winger",
        nationality="Algeria",
        is_public=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    ),
    PlayerModel(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, "player:dz2"),
        first_name="Baghdad",
        last_name="Bounedjah",
        position="Striker",
        nationality="Algeria",
        is_public=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    ),
]


def _is_db_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, SQLAlchemyError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    exc_str = str(exc).lower()
    if any(kw in exc_str for kw in ['connection', 'connect', 'timeout', 'gaierror', 'resolve', 'host', 'network']):
        return True
    return False


@router.post("", response_model=PlayerRead)
async def create_player(
    payload: PlayerCreate,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PlayerModel:
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    player = PlayerModel(**payload.model_dump())
    db.add(player)
    await db.flush()
    return player


@router.get("/{player_id}", response_model=PlayerRead)
async def get_player(
    player_id: str,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PlayerModel:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        player_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"player:{player_id}")
    if db is None:
        for p in _DEMO_PLAYERS:
            if p.id == player_uuid:
                return p
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")
    try:
        player = (await db.execute(select(PlayerModel).options(defer(PlayerModel.verification_status)).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    except Exception as exc:
        if _is_db_unavailable(exc):
            for p in _DEMO_PLAYERS:
                if p.id == player_uuid:
                    return p
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to query player: {exc}")
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")
    return player


@router.patch("/{player_id}", response_model=PlayerRead)
async def update_player(
    player_id: str,
    payload: PlayerUpdate,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PlayerModel:
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    try:
        player = (await db.execute(select(PlayerModel).where(PlayerModel.id == _uuid(player_id)))).scalar_one_or_none()
    except Exception as exc:
        if _is_db_unavailable(exc):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to query player: {exc}")
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(player, field, value)
    await db.flush()
    return player


@router.get("", response_model=list[PlayerRead])
async def list_players(
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    q: str | None = Query(default=None),
    position: str | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[PlayerModel]:
    if db is None:
        return _DEMO_PLAYERS[:limit]
    try:
        query = select(PlayerModel).options(defer(PlayerModel.verification_status))
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
    except Exception as exc:
        if _is_db_unavailable(exc):
            return _DEMO_PLAYERS[:limit]
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list players: {exc}")
