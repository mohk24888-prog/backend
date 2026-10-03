from __future__ import annotations

import socket
import uuid
from typing import Annotated, Optional

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
from core.config import settings
from db.models import Analysis as AnalysisModel, AnalysisJob as AnalysisJobModel, Player as PlayerModel, User as UserModel, HeatmapPoint as HeatmapPointModel
from workers.tasks import analyze_video

router = APIRouter()


def _redis_reachable() -> bool:
    try:
        host = "localhost"
        port = 6379
        from urllib.parse import urlparse
        parsed = urlparse(settings.celery_broker_url)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


_DEMO_ANALYSIS = {
    "id": str(uuid.uuid4()),
    "job_id": str(uuid.uuid4()),
    "player_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "player:dz1")),
    "video_id": str(uuid.uuid4()),
    "match_name": "Demo Match",
    "date": None,
    "overall_rating": 87.5,
    "technical": 88.0,
    "tactical": 86.5,
    "physical": 89.0,
    "mental": 87.0,
    "summary": "Exceptional match performance with outstanding positioning and decision making.",
    "strengths": ["Speed", "Dribbling", "Vision"],
    "development_areas": ["Defensive positioning", "Aerial duels"],
    "overlay_data": {
        "frames": [
            {
                "t": 0.0,
                "players": [
                    {"x": 640, "y": 360, "bbox": [620, 340, 660, 380], "is_subject": True, "track_id": 1, "team_id": 0}
                ],
                "ball": {"x": 645, "y": 365},
                "event": None,
                "speed_kmh": 28.5,
            },
            {
                "t": 1.0,
                "players": [
                    {"x": 650, "y": 355, "bbox": [630, 335, 670, 375], "is_subject": True, "track_id": 1, "team_id": 0}
                ],
                "ball": {"x": 655, "y": 360},
                "event": {"type": "Progressive Carry", "player": "R. Mahrez", "minute": 23},
                "speed_kmh": 31.2,
            },
        ]
    },
    "simulation_data": {"scenario": "1v1", "success_rate": 0.78},
    "video_url": "/static/dz1/demo_match.mp4",
    "analysis_duration_s": 12.5,
    "cv_repo_used": "demo-fallback",
    "subject_track_id": 1,
    "selection_method": "manual",
    "pipeline_warnings": None,
}


@router.post("", response_model=AnalysisRead)
async def create_analysis(
    payload: AnalysisCreate,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisModel:
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")

    try:
        player_uuid = uuid.UUID(str(payload.player_id))
    except (ValueError, TypeError):
        player_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"player:{payload.player_id}")

    player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    if not player:
        player = PlayerModel(
            id=player_uuid,
            first_name="Demo",
            last_name=str(payload.player_id),
            position="Forward",
            nationality="Algeria",
        )
        db.add(player)
        await db.flush()

    job = AnalysisJobModel(
        video_id=payload.video_id,
        player_id=player_uuid,
        status="queued",
    )
    db.add(job)
    await db.flush()

    analysis = AnalysisModel(
        player_id=player_uuid,
        job_id=job.id,
        match_name=payload.match_name,
    )
    db.add(analysis)
    await db.flush()

    import logging
    _logger = logging.getLogger(__name__)

    if _redis_reachable():
        try:
            analyze_video.apply_async(args=[str(job.id)], queue="footiq-analysis", retry=False)
        except Exception as exc:
            job.status = "failed"
            job.error = f"Failed to queue analysis: {exc}"
            db.add(job)
            await db.flush()
            _logger.warning("Celery task queue failed: {}", exc)
    else:
        job.status = "failed"
        job.error = "Redis broker not reachable - analysis task not queued"
        db.add(job)
        await db.flush()
        _logger.warning("Redis not available, skipping analysis task queue")

    await db.flush()
    await db.commit()
    await db.refresh(analysis)
    return analysis


@router.get("/{analysis_id}", response_model=AnalysisRead)
async def get_analysis(
    analysis_id: str,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisModel:
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_uuid))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


_DEMO_ANALYSIS = {
    "id": str(uuid.uuid4()),
    "job_id": str(uuid.uuid4()),
    "player_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "player:dz1")),
    "video_id": str(uuid.uuid4()),
    "match_name": "Demo Match",
    "date": None,
    "overall_rating": 87.5,
    "technical": 88.0,
    "tactical": 86.5,
    "physical": 89.0,
    "mental": 87.0,
    "summary": "Exceptional match performance with outstanding positioning and decision making.",
    "strengths": ["Speed", "Dribbling", "Vision"],
    "development_areas": ["Defensive positioning", "Aerial duels"],
    "overlay_data": {
        "frames": [
            {
                "t": 0.0,
                "players": [
                    {"x": 640, "y": 360, "bbox": [620, 340, 660, 380], "is_subject": True, "track_id": 1, "team_id": 0}
                ],
                "ball": {"x": 645, "y": 365},
                "event": None,
                "speed_kmh": 28.5,
            },
            {
                "t": 1.0,
                "players": [
                    {"x": 650, "y": 355, "bbox": [630, 335, 670, 375], "is_subject": True, "track_id": 1, "team_id": 0}
                ],
                "ball": {"x": 655, "y": 360},
                "event": {"type": "Progressive Carry", "player": "R. Mahrez", "minute": 23},
                "speed_kmh": 31.2,
            },
        ]
    },
    "simulation_data": {"scenario": "1v1", "success_rate": 0.78},
    "video_url": "/static/dz1/demo_match.mp4",
    "analysis_duration_s": 12.5,
    "cv_repo_used": "demo-fallback",
    "subject_track_id": 1,
    "selection_method": "manual",
    "pipeline_warnings": None,
}


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError):
        return False


@router.get("/{analysis_id}/overlay", response_model=AnalysisOverlayResponse)
async def get_analysis_overlay(
    analysis_id: str,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    if db is None:
        return {
            "analysis_id": uuid.UUID(analysis_id) if _is_valid_uuid(analysis_id) else uuid.uuid4(),
            "overlay_data": _DEMO_ANALYSIS["overlay_data"],
            "subject_track_id": _DEMO_ANALYSIS["subject_track_id"],
        }
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_uuid))).scalar_one_or_none()
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
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    if db is None:
        return {
            "analysis_id": uuid.UUID(analysis_id) if _is_valid_uuid(analysis_id) else uuid.uuid4(),
            "simulation": _DEMO_ANALYSIS["simulation_data"],
        }
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_uuid))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return {
        "analysis_id": analysis.id,
        "simulation": analysis.simulation_data,
    }


@router.get("/{analysis_id}/heatmap", response_model=AnalysisHeatmapResponse)
async def get_analysis_heatmap(
    analysis_id: str,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    if db is None:
        return {
            "analysis_id": uuid.UUID(analysis_id) if _is_valid_uuid(analysis_id) else uuid.uuid4(),
            "points": [],
            "image_url": None,
        }
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_uuid))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    points = (await db.execute(select(HeatmapPointModel).where(HeatmapPointModel.analysis_id == analysis_uuid))).scalars().all()

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
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> dict:
    if db is None:
        return {
            "analysis_id": uuid.UUID(analysis_id) if _is_valid_uuid(analysis_id) else uuid.uuid4(),
            "points": [],
            "image_url": None,
        }
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_uuid))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    return {
        "analysis_id": analysis.id,
        "points": [],
        "image_url": analysis.video_url,
    }


@router.get("/jobs/{job_id}", response_model=AnalysisJobRead)
async def get_analysis_job(
    job_id: str,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisJobModel:
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    try:
        job_uuid = uuid.UUID(str(job_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid job ID")
    job = (await db.execute(select(AnalysisJobModel).where(AnalysisJobModel.id == job_uuid))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    return job
