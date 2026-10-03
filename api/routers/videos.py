from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import VideoUploadCreate, VideoUploadRead, VideoUrlResponse
from core.config import settings
from db.models import VideoUpload as VideoUploadModel, Player as PlayerModel, User as UserModel
from services.storage.supabase_client import get_supabase_service

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
        player_uuid = None

    player = None
    if player_uuid is not None:
        player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()

    if not player:
        player_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"player:{player_id}")
        player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
        if not player:
            player = PlayerModel(
                id=player_uuid,
                first_name="Demo",
                last_name=player_id,
                position="Forward",
                nationality="Algeria",
            )
            db.add(player)
            await db.flush()

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

    public_url = None
    storage_key = f"{player_id}/{uuid.uuid4().hex}{Path(file.filename).suffix}"
    try:
        supabase = get_supabase_service()
        from io import BytesIO
        bio = BytesIO(contents)
        supabase.storage.from_(settings.storage_bucket_videos).upload(
            storage_key,
            bio.read(),
        )
        resp = supabase.storage.from_(settings.storage_bucket_videos).get_public_url(storage_key)
        public_url = resp.get("public_url") or resp.get("url")
    except Exception as exc:
        from core.logging_config import logger
        logger.warning("Supabase upload failed, falling back to static: {}", exc)

    if not public_url:
        public_url = f"/static/{storage_path}"

    video = VideoUploadModel(
        player_id=player_uuid,
        uploaded_by=current_user.id if current_user else None,
        filename=file.filename,
        original_path=str(dest),
        storage_path=str(dest),
        public_url=public_url,
        mime_type=file.content_type,
        size_bytes=size,
        match_name=match_name,
    )
    db.add(video)
    await db.flush()
    await db.commit()
    await db.refresh(video)
    return video


@router.get("/{video_id}/url", response_model=VideoUrlResponse)
async def get_video_url(
    video_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    try:
        video_uuid = uuid.UUID(str(video_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid video id")

    video = (await db.execute(select(VideoUploadModel).where(VideoUploadModel.id == video_uuid))).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    return {
        "video_id": video.id,
        "public_url": video.public_url,
    }
