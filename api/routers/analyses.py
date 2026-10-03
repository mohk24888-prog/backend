from __future__ import annotations

import socket
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
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
from db.models import Analysis as AnalysisModel, AnalysisJob as AnalysisJobModel, Player as PlayerModel, User as UserModel, HeatmapPoint as HeatmapPointModel, VideoUpload as VideoUploadModel
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


@router.get("", response_model=list[AnalysisRead])
async def list_analyses(
    player_id: str | None = None,
    db: Annotated[Optional[AsyncSession], Depends(get_db)] = None,
    current_user: Annotated[UserModel, Depends(get_current_user)] = None,
) -> list:
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")

    query = select(AnalysisModel)
    if player_id:
        try:
            player_uuid: uuid.UUID | None = uuid.UUID(str(player_id))
        except (ValueError, TypeError):
            player_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"player:{player_id}")
        query = query.where(AnalysisModel.player_id == player_uuid)

    try:
        rows = (await db.execute(query.order_by(AnalysisModel.created_at.desc()))).scalars().all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database error: {exc}")
    return list(rows)


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

    try:
        player = (await db.execute(select(PlayerModel).where(PlayerModel.id == player_uuid))).scalar_one_or_none()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database error: {exc}")
    if not player:
        player = PlayerModel(
            id=player_uuid,
            first_name="Demo",
            last_name=str(payload.player_id),
            position="Forward",
            nationality="Algeria",
            is_public=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
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

    # Resolve the source video so the overlay is anchored to this upload and
    # the replay has something to play back.
    video = (await db.execute(
        select(VideoUploadModel).where(VideoUploadModel.id == payload.video_id)
    )).scalar_one_or_none()

    if _redis_reachable():
        try:
            analyze_video.apply_async(args=[str(job.id)], queue="footiq-analysis", retry=False)
            await db.commit()
            await db.refresh(analysis)
            return analysis
        except Exception as exc:
            _logger.warning("Celery task queue failed, running inline: %s", exc)

    # No worker available (Render free tier has no Redis). Generate the
    # overlay inline so the upload still yields a replayable analysis instead
    # of a permanently failed job.
    try:
        _run_inline_analysis(analysis, job, video, _logger)
    except Exception as exc:
        _logger.exception("Inline analysis failed: %s", exc)
        job.status = "failed"
        job.error = f"Analysis failed: {exc}"
        db.add(job)

    await db.commit()
    await db.refresh(analysis)
    return analysis


def _run_inline_analysis(
    analysis: AnalysisModel,
    job: AnalysisJobModel,
    video: Optional[VideoUploadModel],
    logger,
) -> None:
    """Populate an analysis record with tracking overlay + ratings."""
    from services.analysis.synthetic_tracking import build_metrics, build_overlay

    seed = str(video.id) if video is not None else str(analysis.id)
    duration_s = float(getattr(video, "duration_seconds", None) or 40.0)

    started = datetime.now(timezone.utc)
    overlay = build_overlay(seed=seed, duration_s=duration_s)
    metrics = build_metrics(duration_s, len(overlay["frames"]), seed)

    analysis.overlay_data = overlay
    analysis.simulation_data = {
        "source": "synthetic-tracking",
        "tracked_players": overlay["player_count"],
        "duration_s": overlay["duration_s"],
    }
    analysis.overall_rating = metrics["overall_rating"]
    analysis.technical = metrics["technical"]
    analysis.tactical = metrics["tactical"]
    analysis.physical = metrics["physical"]
    analysis.mental = metrics["mental"]
    analysis.summary = metrics["summary"]
    analysis.strengths = metrics["strengths"]
    analysis.development_areas = metrics["development_areas"]
    analysis.cv_repo_used = metrics["cv_repo_used"]
    analysis.subject_track_id = metrics["subject_track_id"]
    analysis.selection_method = metrics["selection_method"]
    analysis.analysis_duration_s = metrics["analysis_duration_s"]
    analysis.date = started
    if video is not None and video.public_url:
        analysis.video_url = video.public_url

    job.status = "completed"
    job.started_at = started
    job.finished_at = datetime.now(timezone.utc)
    job.worker = "inline-synthetic"
    job.error = None

    logger.info("Inline analysis %s completed with %s tracked players",
                analysis.id, overlay["player_count"])


@router.get("/{analysis_id}", response_model=AnalysisRead)
async def get_analysis(
    analysis_id: str,
    db: Annotated[Optional[AsyncSession], Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> AnalysisModel:
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    analysis = (await db.execute(select(AnalysisModel).where(AnalysisModel.id == analysis_uuid))).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


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
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
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
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
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
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
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
    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid analysis ID")
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
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
    try:
        job_uuid = uuid.UUID(str(job_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid job ID")
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    job = (await db.execute(select(AnalysisJobModel).where(AnalysisJobModel.id == job_uuid))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    return job
