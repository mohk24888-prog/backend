from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import (
    AnalysisTestMappingRead,
    TestSessionCreate,
    TestSessionRead,
    TestSessionTestTypeCreate,
    TestSessionTestTypeRead,
    TestType,
)
from db.models import Analysis as AnalysisModel, Player as PlayerModel, TestSession as TestSessionModel, TestSessionTestType as TestSessionTestTypeModel, User as UserModel

router = APIRouter()


@router.post("", response_model=TestSessionRead)
async def create_test_session(
    payload: TestSessionCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> TestSessionModel:
    try:
        player_uuid = uuid.UUID(str(payload.player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    session = TestSessionModel(
        player_id=player_uuid,
        session_date=payload.session_date,
        location=payload.location,
        notes=payload.notes,
    )
    db.add(session)
    await db.flush()
    return session


@router.get("/{session_id}", response_model=TestSessionRead)
async def get_test_session(
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> TestSessionModel:
    try:
        session_uuid = uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id")

    session = (await db.execute(select(TestSessionModel).where(TestSessionModel.id == session_uuid))).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test session not found")
    return session


@router.post("/{session_id}/tests", response_model=TestSessionTestTypeRead)
async def add_test_to_session(
    session_id: str,
    payload: TestSessionTestTypeCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> TestSessionTestTypeModel:
    try:
        session_uuid = uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id")

    session = (await db.execute(select(TestSessionModel).where(TestSessionModel.id == session_uuid))).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test session not found")

    test_type = TestSessionTestTypeModel(
        test_session_id=session_uuid,
        test_type=payload.test_type,
        config=payload.config,
    )
    db.add(test_type)
    await db.flush()
    return test_type


@router.get("/{session_id}/tests", response_model=list[TestSessionTestTypeRead])
async def get_session_tests(
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> list[TestSessionTestTypeModel]:
    try:
        session_uuid = uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id")

    session = (await db.execute(select(TestSessionModel).where(TestSessionModel.id == session_uuid))).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test session not found")
    result = await db.execute(select(TestSessionTestTypeModel).where(TestSessionTestTypeModel.test_session_id == session_uuid))
    return result.scalars().all()


@router.get("/player/{player_id}", response_model=list[TestSessionRead])
async def get_player_sessions(
    player_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> list[TestSessionModel]:
    try:
        player_uuid = uuid.UUID(str(player_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid player id")

    result = await db.execute(select(TestSessionModel).where(TestSessionModel.player_id == player_uuid).order_by(TestSessionModel.created_at.desc()))
    return result.scalars().all()