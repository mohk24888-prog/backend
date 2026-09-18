from __future__ import annotations

import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import NotificationCreate, NotificationRead
from db.models import Notification as NotificationModel, User as UserModel

router = APIRouter()


@router.get("", response_model=List[NotificationRead])
async def list_notifications(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    unread_only: bool = False,
) -> List[NotificationModel]:
    query = select(NotificationModel).where(NotificationModel.user_id == current_user.id)
    if unread_only:
        query = query.where(NotificationModel.read.is_(False))
    result = await db.execute(query.order_by(NotificationModel.created_at.desc()))
    return result.scalars().all()


@router.patch("/{notification_id}/read", response_model=NotificationRead)
async def mark_notification_read(
    notification_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> NotificationModel:
    try:
        notification_uuid = uuid.UUID(str(notification_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notification id")

    notification = (
        await db.execute(
            select(NotificationModel).where(
                NotificationModel.id == notification_uuid,
                NotificationModel.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not notification:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    notification.read = True
    await db.flush()
    return notification
