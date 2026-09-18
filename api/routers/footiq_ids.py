from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from services.footiq_id import generate_footiq_id
from db.models import FootIQPlayerId as FootIQPlayerIdModel, Player as PlayerModel, User as UserModel
from api.schemas.footiq import FootIQPlayerIdRead

router = APIRouter()


@router.post("/{player_id}/footiq-id", response_model=FootIQPlayerIdRead)
async def issue_footiq_id(
    player_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> FootIQPlayerIdModel:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    existing = (await db.execute(select(FootIQPlayerIdModel).where(FootIQPlayerIdModel.player_id == player_uuid))).scalar_one_or_none()
    if existing:
        return existing

    footiq_id = generate_footiq_id()
    record = FootIQPlayerIdModel(
        player_id=player_uuid,
        footiq_id=footiq_id,
        issued_at=None,
        status="active",
    )
    db.add(record)
    await db.flush()
    return record


@router.get("/{player_id}/footiq-id", response_model=FootIQPlayerIdRead)
async def get_footiq_id(
    player_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> FootIQPlayerIdModel:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    result = await db.execute(select(FootIQPlayerIdModel).where(FootIQPlayerIdModel.player_id == player_uuid))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="FootIQ ID not issued")
    return record