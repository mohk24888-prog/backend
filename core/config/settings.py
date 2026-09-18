from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="local")
    log_level: str = Field(default="INFO")

    supabase_url: str = Field(default="http://localhost:54321")
    supabase_anon_key: str = Field(default="")
    supabase_service_role_key: str = Field(default="")
    supabase_jwt_secret: str = Field(default="")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:54322/postgres"
    )
    database_url_sync: Optional[str] = Field(default=None)

    redis_url: str = Field(default="redis://localhost:6379/0")
    celery_broker_url: str = Field(default="redis://localhost:6379/0")
    celery_result_backend: str = Field(default="redis://localhost:6379/1")

    storage_bucket_player_photos: str = Field(default="player-photos")
    storage_bucket_videos: str = Field(default="videos")
    storage_bucket_reports: str = Field(default="reports")
    storage_bucket_thumbnails: str = Field(default="thumbnails")

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080,http://localhost:5555"
    )

    api_v1_str: str = Field(default="/api/v1")
    project_name: str = Field(default="FootIQ API")
    version: str = Field(default="0.1.0")

    device: str = Field(default="auto")
    yolo_model: str = Field(default="yolov10n.pt")
    confidence_threshold: float = Field(default=0.35)
    tracker: str = Field(default="bytetrack")
    frame_skip: int = Field(default=2)
    max_video_duration: int = Field(default=600)
    max_video_size_mb: int = Field(default=500)
    allowed_video_extensions: str = Field(default="mp4,mov,avi,mkv")

    raw_dir: Path = Field(default=Path("/tmp/footiq/raw"))
    processed_dir: Path = Field(default=Path("/tmp/footiq/processed"))


settings = Settings()

if settings.database_url_sync is None:
    object.__setattr__(settings, "database_url_sync", settings.database_url.replace("+asyncpg", ""))
