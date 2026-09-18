from __future__ import annotations

import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import MessageCreate, MessageRead, ConversationRead
from db.models import Conversation as ConversationModel, ConversationParticipant as ParticipantModel, Message as MessageModel, User as UserModel

router = APIRouter()


@router.get("", response_model=List[ConversationRead])
async def list_conversations(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> List[ConversationModel]:
    sub = (
        select(ConversationModel)
        .join(ParticipantModel, ParticipantModel.conversation_id == ConversationModel.id)
        .where(ParticipantModel.user_id == current_user.id)
    )
    result = await db.execute(sub)
    return result.scalars().all()


@router.post("/{conversation_id}/messages", response_model=MessageRead)
async def send_message(
    conversation_id: str,
    payload: MessageCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> MessageModel:
    try:
        conversation_uuid = uuid.UUID(str(conversation_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation id")

    participant = (
        await db.execute(
            select(ParticipantModel).where(
                ParticipantModel.conversation_id == conversation_uuid,
                ParticipantModel.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a participant")

    message = MessageModel(conversation_id=conversation_uuid, sender_id=current_user.id, **payload.model_dump())
    db.add(message)
    await db.flush()
    return message


@router.get("/{conversation_id}/messages", response_model=List[MessageRead])
async def list_messages(
    conversation_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    limit: int = 50,
) -> List[MessageModel]:
    try:
        conversation_uuid = uuid.UUID(str(conversation_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation id")

    participant = (
        await db.execute(
            select(ParticipantModel).where(
                ParticipantModel.conversation_id == conversation_uuid,
                ParticipantModel.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a participant")

    result = await db.execute(
        select(MessageModel)
        .where(MessageModel.conversation_id == conversation_uuid)
        .order_by(MessageModel.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))
