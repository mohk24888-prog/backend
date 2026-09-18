from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.config import settings
from core.logging_config import setup_logging, log_request
from db.session import init_db

from api.routers import (
    auth,
    users,
    players,
    videos,
    analyses,
    test_sessions,
    calibration,
    footiq_ids,
    performance,
    cv,
)

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("FootIQ minimal API starting")
    await init_db()
    yield
    logger.info("FootIQ minimal API shutting down")


app = FastAPI(
    title=settings.project_name,
    description="FootIQ standardized one-player testing API",
    version=settings.version,
    lifespan=lifespan,
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
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
app.include_router(videos.router, prefix=f"{settings.api_v1_str}/videos", tags=["Videos"])
app.include_router(analyses.router, prefix=f"{settings.api_v1_str}/analyses", tags=["Analyses"])
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
