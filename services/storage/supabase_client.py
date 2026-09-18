from __future__ import annotations

from typing import Optional

from supabase import create_client, Client
from core.config import settings


def get_supabase() -> Client:
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError("Supabase credentials are not configured")
    return create_client(settings.supabase_url, settings.supabase_anon_key)


def get_supabase_service() -> Client:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("Supabase service credentials are not configured")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
