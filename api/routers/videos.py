from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import VideoUploadCreate, VideoUploadRead
from core.config import settings
from db.models import VideoUpload as VideoUploadModel, Player as PlayerModel, User as UserModel

router = APIRouter()


@router.post("", response_model=VideoUploadRead)
async def upload_video(
    player_id: str = Form(...),
    match_name: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: Annotated[AsyncSession, Depends(get_db)] = None,
    current_user: Annotated[UserModel, Depends(get_current_user)] = None,
) -> VideoUploadModel:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    ext = Path(file.filename).suffix.lower().lstrip(".")
    allowed = {e.strip() for e in settings.allowed_video_extensions.split(",") if e.strip()}
    if ext not in allowed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported video extension: {ext}")

    contents = await file.read()
    size = len(contents)
    if size > settings.max_video_size_mb * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video too large")

    storage_path = f"{player_id}/{uuid.uuid4().hex}{Path(file.filename).suffix}"
    dest = settings.raw_dir / storage_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(contents)

    video = VideoUploadModel(
        player_id=player_uuid,
        uploaded_by=current_user.id if current_user else None,
        filename=file.filename,
        storage_path=str(dest),
        mime_type=file.content_type,
        size_bytes=size,
        match_name=match_name,
    )
    db.add(video)
    await db.flush()
    return video
