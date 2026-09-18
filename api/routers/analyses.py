from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import AnalysisCreate, AnalysisRead, AnalysisJobRead
from db.models import Analysis as AnalysisModel, AnalysisJob as AnalysisJobModel, Player as PlayerModel, User as UserModel

router = APIRouter()


@router.post("", response_model=AnalysisRead)
async def create_analysis(
    payload: AnalysisCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisModel:
    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == payload.player_id))).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    analysis = AnalysisModel(player_id=payload.player_id, match_name=payload.match_name)
    db.add(analysis)
    await db.flush()
    return analysis


@router.get("/{analysis_id}", response_model=AnalysisRead)
async def get_analysis(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisModel:
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_id))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


@router.get("/jobs/{job_id}", response_model=AnalysisJobRead)
async def get_analysis_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisJobModel:
    job = (await db.execute(select(AnalysisJobModel).where(AnalysisJobModel.id == job_id))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    return job
