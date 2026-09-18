from __future__ import annotations

import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import PlayerRead
from db.models import Club as ClubModel, Player as PlayerModel, User as UserModel

router = APIRouter()


@router.get("/{club_id}/players", response_model=List[PlayerRead])
async def list_club_players(
    club_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> List[PlayerModel]:
    try:
        club_uuid = uuid.UUID(str(club_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid club id")

    club = (await db.execute(select(ClubModel).where(ClubModel.id == club_uuid))).scalar_one_or_none()
    if not club:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Club not found")
    return []
