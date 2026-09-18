from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import MessageCreate, MessageRead
from db.models import Conversation, ConversationParticipant, Message, User as UserModel

router = APIRouter()


@router.post("/{conversation_id}/messages", response_model=MessageRead)
async def send_message(
    conversation_id: str,
    payload: MessageCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> Message:
    try:
        conversation_uuid = uuid.UUID(str(conversation_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation id")

    participant = (
        await db.execute(
            select(ConversationParticipant)
            .where(
                ConversationParticipant.conversation_id == conversation_uuid,
                ConversationParticipant.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a participant")

    message = Message(
        conversation_id=conversation_uuid,
        sender_id=current_user.id,
        **payload.model_dump(),
    )
    db.add(message)
    await db.flush()
    return message


@router.get("/{conversation_id}/messages", response_model=list[MessageRead])
async def list_messages(
    conversation_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    limit: int = Query(default=50, le=200),
) -> list[Message]:
    try:
        conversation_uuid = uuid.UUID(str(conversation_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation id")

    participant = (
        await db.execute(
            select(ConversationParticipant)
            .where(
                ConversationParticipant.conversation_id == conversation_uuid,
                ConversationParticipant.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a participant")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_uuid)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))
