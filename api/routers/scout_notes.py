from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import ScoutNoteCreate, ScoutNoteRead
from db.models import ScoutNote as ScoutNoteModel, User as UserModel

router = APIRouter()


@router.post("", response_model=ScoutNoteRead)
async def create_scout_note(
    payload: ScoutNoteCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> ScoutNoteModel:
    player_uuid = uuid.UUID(str(payload.player_id)) if payload.player_id else None
    note = ScoutNoteModel(
        author_user_id=current_user.id,
        player_id=player_uuid,
        content=payload.content,
        is_private=payload.is_private,
    )
    db.add(note)
    await db.flush()
    return note


@router.get("", response_model=list[ScoutNoteRead])
async def list_scout_notes(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    player_id: str | None = None,
) -> list[ScoutNoteModel]:
    query = select(ScoutNoteModel).where(ScoutNoteModel.author_user_id == current_user.id)
    if player_id:
        try:
            query = query.where(ScoutNoteModel.player_id == uuid.UUID(str(player_id)))
        except (ValueError, TypeError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")
    result = await db.execute(query.order_by(ScoutNoteModel.created_at.desc()))
    return result.scalars().all()
