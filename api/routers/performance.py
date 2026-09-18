from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from db.models import Player as PlayerModel, TestSession as TestSessionModel, TestSessionTestType as TestSessionTestTypeModel, User as UserModel
from api.schemas.footiq import TestSessionRead, TestSessionTestTypeRead

router = APIRouter()


@router.get("/player/{player_id}/performance")
async def get_player_performance(
    player_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    sessions = (await db.execute(select(TestSessionModel).where(TestSessionModel.player_id == player_uuid).order_by(TestSessionModel.created_at.desc()))).scalars().all()

    session_data = []
    for session in sessions:
        tests = (await db.execute(select(TestSessionTestTypeModel).where(TestSessionTestTypeModel.test_session_id == session.id))).scalars().all()
        test_types = [t.test_type.value for t in tests]
        session_data.append({
            "session_id": str(session.id),
            "session_date": session.session_date,
            "location": session.location,
            "notes": session.notes,
            "test_types": test_types,
        })

    return {
        "player_id": player_id,
        "player_name": f"{player.first_name} {player.last_name}",
        "position": player.position,
        "nationality": player.nationality,
        "total_sessions": len(sessions),
        "sessions": session_data,
    }