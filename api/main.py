from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

from api.routers import (
    auth,
    users,
    players,
    clubs,
    academies,
    agents,
    videos,
    analyses,
    discover,
    watchlists,
    scout_notes,
    offers,
    conversations,
    messages,
    notifications,
    metrics,
    test_sessions,
    calibration,
    footiq_ids,
    performance,
    cv,
)
from core.config import settings
from core.logging_config import setup_logging, log_request
from db.session import init_db, async_session_factory, _active_url

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("FootIQ API starting up")
    await init_db()
    await _seed_demo_data()
    yield
    logger.info("FootIQ API shutting down")


async def _seed_demo_data():
    """Seed demo players, video, and analysis so the Flutter app has real data for the demo."""
    try:
        from db.session import async_session_factory
        from db.models import Player as PlayerModel, VideoUpload as VideoUploadModel, AnalysisJob as AnalysisJobModel, Analysis as AnalysisModel
        from db.models import AnalysisStatus

        demo_players = [
            {"id_str": "dz1", "first_name": "Riyad", "last_name": "Mahrez", "position": "Right Winger", "nationality": "Algeria"},
            {"id_str": "dz2", "first_name": "Baghdad", "last_name": "Bounedjah", "position": "Striker", "nationality": "Algeria"},
        ]

        async with async_session_factory() as session:
            player_ids = {}
            for p in demo_players:
                pid = uuid.uuid5(uuid.NAMESPACE_DNS, f"player:{p['id_str']}")
                player_ids[p['id_str']] = pid
                existing = await session.get(PlayerModel, pid)
                if not existing:
                    player = PlayerModel(
                        id=pid,
                        first_name=p["first_name"],
                        last_name=p["last_name"],
                        position=p["position"],
                        nationality=p["nationality"],
                    )
                    session.add(player)

            # Seed one completed demo analysis for dz1 if none exists
            dz1_id = player_ids["dz1"]
            existing_analysis = (await session.execute(
                select(AnalysisModel).where(AnalysisModel.player_id == dz1_id)
            )).scalar_one_or_none()

            if not existing_analysis:
                from datetime import datetime, timezone
                from pathlib import Path

                raw_dir = Path(settings.raw_dir)
                raw_dir.mkdir(parents=True, exist_ok=True)
                demo_video_path = raw_dir / "dz1" / "demo_match.mp4"
                demo_video_path.parent.mkdir(parents=True, exist_ok=True)
                demo_video_path.write_bytes(b"demo video content")

                video = VideoUploadModel(
                    player_id=dz1_id,
                    filename="demo_match.mp4",
                    storage_path=str(demo_video_path),
                    public_url=f"/static/dz1/demo_match.mp4",
                    mime_type="video/mp4",
                    size_bytes=123456,
                    match_name="Demo Match",
                )
                session.add(video)
                await session.flush()

                job = AnalysisJobModel(
                    video_id=video.id,
                    player_id=dz1_id,
                    status=AnalysisStatus.completed,
                    worker="local-seed",
                    started_at=datetime.now(timezone.utc),
                    finished_at=datetime.now(timezone.utc),
                )
                session.add(job)
                await session.flush()

                analysis = AnalysisModel(
                    player_id=dz1_id,
                    job_id=job.id,
                    video_id=video.id,
                    match_name="Demo Match",
                    date=datetime.now(timezone.utc),
                    overall_rating=87.5,
                    technical=88.0,
                    tactical=86.5,
                    physical=89.0,
                    mental=87.0,
                    summary="Exceptional match performance with outstanding positioning and decision making.",
                    strengths=["Speed", "Dribbling", "Vision"],
                    development_areas=["Defensive positioning", "Aerial duels"],
                    overlay_data={
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
                    simulation_data={"scenario": "1v1", "success_rate": 0.78},
                    video_url=f"/static/dz1/demo_match.mp4",
                    analysis_duration_s=12.5,
                    cv_repo_used="local-seed",
                    subject_track_id=1,
                    selection_method="manual",
                )
                session.add(analysis)
                await session.commit()
                logger.info("Demo analysis seeded for dz1")

            await session.commit()
            logger.info("Demo players seeded successfully")
    except Exception as e:
        logger.warning("Demo seeding failed: {}", e)


app = FastAPI(
    title=settings.project_name,
    description="FootIQ Football Intelligence Platform API",
    version=settings.version,
    lifespan=lifespan,
)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: Exception):
    logger.warning("Database error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database unavailable", "path": str(request.url.path)},
    )


@app.exception_handler(OSError)
async def os_exception_handler(request: Request, exc: Exception):
    exc_str = str(exc).lower()
    if any(kw in exc_str for kw in ['connection', 'connect', 'timeout', 'gaierror', 'resolve', 'database']):
        logger.warning("OS/database error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Database unavailable", "path": str(request.url.path)},
        )
    logger.error("Unhandled OS error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)[:200], "path": str(request.url.path)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    exc_str = str(exc).lower()
    if any(kw in exc_str for kw in ['connection', 'connect', 'timeout', 'gaierror', 'resolve', 'database']):
        logger.warning("Database-related error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Database unavailable", "path": str(request.url.path)},
        )
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)[:200], "path": str(request.url.path)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        log_request(request)
        response: Response = await call_next(request)
        return response


app.add_middleware(RequestLoggingMiddleware)

os.makedirs(settings.raw_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(settings.raw_dir)), name="static")

app.include_router(auth.router, prefix=f"{settings.api_v1_str}/auth", tags=["Auth"])
app.include_router(users.router, prefix=f"{settings.api_v1_str}/users", tags=["Users"])
app.include_router(players.router, prefix=f"{settings.api_v1_str}/players", tags=["Players"])
app.include_router(clubs.router, prefix=f"{settings.api_v1_str}/clubs", tags=["Clubs"])
app.include_router(academies.router, prefix=f"{settings.api_v1_str}/academies", tags=["Academies"])
app.include_router(agents.router, prefix=f"{settings.api_v1_str}/agents", tags=["Agents"])
app.include_router(videos.router, prefix=f"{settings.api_v1_str}/videos", tags=["Videos"])
app.include_router(analyses.router, prefix=f"{settings.api_v1_str}/analyses", tags=["Analyses"])
app.include_router(discover.router, prefix=f"{settings.api_v1_str}/discover", tags=["Discover"])
app.include_router(watchlists.router, prefix=f"{settings.api_v1_str}/watchlists", tags=["Watchlists"])
app.include_router(scout_notes.router, prefix=f"{settings.api_v1_str}/scout-notes", tags=["Scout Notes"])
app.include_router(offers.router, prefix=f"{settings.api_v1_str}/offers", tags=["Offers"])
app.include_router(conversations.router, prefix=f"{settings.api_v1_str}/conversations", tags=["Conversations"])
app.include_router(messages.router, prefix=f"{settings.api_v1_str}/messages", tags=["Messages"])
app.include_router(notifications.router, prefix=f"{settings.api_v1_str}/notifications", tags=["Notifications"])
app.include_router(metrics.router, prefix=f"{settings.api_v1_str}/metrics", tags=["Metrics"])
app.include_router(test_sessions.router, prefix=f"{settings.api_v1_str}/test-sessions", tags=["Test Sessions"])
app.include_router(calibration.router, prefix=f"{settings.api_v1_str}/calibration", tags=["Calibration"])
app.include_router(footiq_ids.router, prefix=f"{settings.api_v1_str}/footiq-ids", tags=["FootIQ IDs"])
app.include_router(performance.router, prefix=f"{settings.api_v1_str}/performance", tags=["Performance"])
app.include_router(cv.router, prefix=f"{settings.api_v1_str}/cv", tags=["CV"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": settings.version, "device": settings.device, "commit": "7e24c71"}


@app.get("/health/ready")
async def health_ready() -> dict:
    return {"status": "ready", "version": settings.version, "commit": "7e24c71"}


@app.get("/health/db")
async def health_db() -> dict:
    """Report whether the configured Postgres is actually reachable.

    init_db() probes the configured DATABASE_URL at startup and falls back to
    a local SQLite file when it is unreachable, so the API can still serve
    requests without Supabase. This endpoint reports which backend is active.
    """
    from sqlalchemy import text

    target = (_active_url or "").split("@")[-1]
    is_fallback = (target or "").startswith("/tmp")
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {
            "database": "ok",
            "backend": "sqlite" if is_fallback else "postgres",
            "target": target,
        }
    except Exception as exc:
        return {
            "database": "unreachable",
            "backend": "sqlite" if is_fallback else "postgres",
            "target": target,
            "error": type(exc).__name__,
            "detail": str(exc)[:200],
        }
