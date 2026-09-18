from __future__ import annotations

from celery import Celery
from core.config import settings

celery_app = Celery(
    "footiq-worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.task_routes = {
    "workers.tasks.analyze_video": {"queue": "analysis"},
}
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.task_track_started = True
celery_app.conf.task_time_limit = 60 * 60
celery_app.conf.task_soft_time_limit = 60 * 55
