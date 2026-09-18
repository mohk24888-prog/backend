from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import CalibrationConfigCreate, CalibrationConfigRead
from db.models import CalibrationConfig as CalibrationConfigModel, Player as PlayerModel, User as UserModel

router = APIRouter()


@router.post("", response_model=CalibrationConfigRead)
async def create_calibration(
    payload: CalibrationConfigCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> CalibrationConfigModel:
    try:
        player_uuid = uuid.UUID(str(payload.player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    config = CalibrationConfigModel(
        player_id=player_uuid,
        camera_position=payload.camera_position,
        camera_height_m=payload.camera_height_m,
        field_coordinates=payload.field_coordinates,
        known_distances=payload.known_distances,
        pitch_dimensions=payload.pitch_dimensions,
        test_markers=payload.test_markers,
        start_point=payload.start_point,
        finish_point=payload.finish_point,
        target_positions=payload.target_positions,
        calibration_status=payload.calibration_status or "NOT_CALIBRATED",
    )
    db.add(config)
    await db.flush()
    return config


@router.get("/player/{player_id}", response_model=CalibrationConfigRead)
async def get_calibration(
    player_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> CalibrationConfigModel:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    result = await db.execute(select(CalibrationConfigModel).where(CalibrationConfigModel.player_id == player_uuid).order_by(CalibrationConfigModel.created_at.desc()))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calibration not found")
    return config