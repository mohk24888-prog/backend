from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import (
    AnalysisCreate,
    AnalysisRead,
    AnalysisJobRead,
    AnalysisOverlayResponse,
    AnalysisSimulationResponse,
    AnalysisHeatmapResponse,
    AnalysisTrajectoryResponse,
)
from db.models import Analysis as AnalysisModel, AnalysisJob as AnalysisJobModel, Player as PlayerModel, User as UserModel, HeatmapPoint as HeatmapPointModel
from workers.tasks import analyze_video

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

    job = AnalysisJobModel(
        video_id=payload.video_id,
        player_id=payload.player_id,
        status="queued",
    )
    db.add(job)
    await db.flush()

    analysis = AnalysisModel(
        player_id=payload.player_id,
        job_id=job.id,
        match_name=payload.match_name,
    )
    db.add(analysis)
    await db.flush()

    try:
        analyze_video.apply_async(args=[str(job.id)], queue="footiq-analysis")
    except Exception as exc:
        job.status = "failed"
        job.error = f"Failed to queue analysis: {exc}"
        db.add(job)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to queue analysis job")

    await db.commit()
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


@router.get("/{analysis_id}/overlay", response_model=AnalysisOverlayResponse)
async def get_analysis_overlay(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_id))).scalar_one_or_none()
    if not analysis or not analysis.overlay_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not ready")
    return {
        "analysis_id": analysis.id,
        "overlay_data": analysis.overlay_data,
        "subject_track_id": analysis.subject_track_id,
    }


@router.get("/{analysis_id}/simulation", response_model=AnalysisSimulationResponse)
async def get_analysis_simulation(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_id))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return {
        "analysis_id": analysis.id,
        "simulation": analysis.simulation_data,
    }


@router.get("/{analysis_id}/heatmap", response_model=AnalysisHeatmapResponse)
async def get_analysis_heatmap(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_id))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    points = (await db.execute(select(HeatmapPointModel).where(HeatmapPointModel.analysis_id == analysis_id))).scalars().all()

    return {
        "analysis_id": analysis.id,
        "points": [
            {
                "id": p.id,
                "analysis_id": p.analysis_id,
                "player_id": p.player_id,
                "x": p.x,
                "y": p.y,
                "intensity": p.intensity,
                "frame_number": p.frame_number,
                "created_at": p.created_at,
            }
            for p in points
        ],
        "image_url": analysis.video_url,
    }


@router.get("/{analysis_id}/trajectory", response_model=AnalysisTrajectoryResponse)
async def get_analysis_trajectory(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_id))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    return {
        "analysis_id": analysis.id,
        "points": [],
        "image_url": None,
    }


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
