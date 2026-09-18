#!/usr/bin/env python3
"""Generate initial Alembic migration from SQLAlchemy models."""
from pathlib import Path
from alembic.command import revision
from alembic import config

ini = Path(__file__).resolve().parents[1] / "db" / "migrations" / "alembic.ini"
cfg = config.Config(str(ini))
cfg.set_main_option("sqlalchemy.url", "postgresql+asyncpg://postgres:postgres@localhost:54322/postgres")
revision(cfg, message="initial schema", autogenerate=True)
