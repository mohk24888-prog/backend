from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

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
from db.session import init_db

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("FootIQ API starting up")
    await init_db()
    yield
    logger.info("FootIQ API shutting down")


app = FastAPI(
    title=settings.project_name,
    description="FootIQ Football Intelligence Platform API",
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
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
    return {"status": "ok", "version": settings.version, "device": settings.device}


@app.get("/health/ready")
async def health_ready() -> dict:
    return {"status": "ready", "version": settings.version}
