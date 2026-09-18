from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
from workers.celery_app import celery_app
from db.session import get_session
from sqlalchemy import select
from db.models import AnalysisJob, AnalysisStatus, Player, VideoUpload, Analysis, HeatmapPoint, MatchAction, PhysicalMetric, TacticalMetric, AnalysisTestMapping, TestSession, TestType, CalibrationConfig
from services.analysis.pipline import (
    DeviceManager,
    PlayerDetector,
    Tracker,
    VideoPreprocessor,
    MetricsCalculator,
)
from services.analysis.analyzers.factory import TestAnalyzerFactory
from core.config import settings


@celery_app.task(name="workers.tasks.analyze_video", bind=True)
def analyze_video(self, job_id: str) -> dict:
    session_gen = get_session()
    session = next(session_gen)
    job = None
    try:
        job = session.get(AnalysisJob, job_id)
        if not job:
            logger.error("Analysis job not found: {}", job_id)
            return {"status": "failed", "error": "job_not_found"}

        job.status = AnalysisStatus.preprocessing
        session.add(job)
        session.commit()

        video = session.get(VideoUpload, job.video_id)
        if not video or not video.storage_path:
            job.status = AnalysisStatus.failed
            job.error = "Video not found or not stored"
            session.add(job)
            session.commit()
            return {"status": "failed", "job_id": job_id, "error": "video_not_found"}

        video_path = Path(video.storage_path)
        if not video_path.exists():
            job.status = AnalysisStatus.failed
            job.error = f"Video file not found: {video_path}"
            session.add(job)
            session.commit()
            return {"status": "failed", "job_id": job_id, "error": "video_not_found"}

        logger.info("Analyzing video: {}", video_path)
        job.status = AnalysisStatus.detecting
        job.worker = DeviceManager.get_device_name()
        session.add(job)
        session.commit()

        preprocessor = VideoPreprocessor()
        frames = preprocessor.extract_frames(video_path, frame_skip=settings.frame_skip)
        detector = PlayerDetector(settings.yolo_model)
        tracker = Tracker()
        all_tracked: list[dict] = []
        for frame_id, frame in enumerate(frames):
            detections = detector.detect(frame)
            tracked = tracker.update(detections)
            for t in tracked:
                t["frame_id"] = frame_id
            all_tracked.extend(tracked)

        test_type = None
        calibration_config_id = None
        if job.test_session_id:
            test_session = session.get(TestSession, job.test_session_id)
            if test_session:
                session_test = session.exec(
                    select(TestSessionTestType).where(TestSessionTestType.test_session_id == test_session.id)
                ).first()
                if session_test:
                    test_type = session_test.test_type
                    mapping = session.exec(
                        select(AnalysisTestMapping).where(AnalysisTestMapping.test_type == test_type.value)
                    ).first()
                    if mapping:
                        calibration_config_id = mapping.calibration_config_id

        if test_type is None:
            physical = MetricsCalculator.physical_metrics(all_tracked)
            tactical = MetricsCalculator.tactical_metrics(all_tracked)
            heatmap_points = MetricsCalculator.heatmap(all_tracked)

            analysis = Analysis(
                player_id=job.player_id,
                match_name=video.match_name,
                date=None,
                summary="Analysis completed via FootIQ pipeline.",
                strengths=["Detected movement patterns"],
                development_areas=["Insufficient data for full assessment"],
            )
            session.add(analysis)
            session.flush()

            for pt in heatmap_points:
                hp = HeatmapPoint(
                    analysis_id=analysis.id,
                    player_id=job.player_id,
                    x=pt["x"],
                    y=pt["y"],
                    intensity=pt["intensity"],
                )
                session.add(hp)

            for i in range(0, len(all_tracked), 5):
                t = all_tracked[i]
                action = MatchAction(
                    analysis_id=analysis.id,
                    player_id=job.player_id,
                    minute=int((t.get("frame_id", 0) / settings.frame_skip) / 15) if t.get("frame_id") is not None else None,
                    type="Movement",
                    description=f"Tracked player at center {t.get('center', [0, 0])}",
                    category="General",
                    success=None,
                )
                session.add(action)

            pm = PhysicalMetric(
                analysis_id=analysis.id,
                player_id=job.player_id,
                **{k: v for k, v in physical.items() if k != "confidence"},
            )
            session.add(pm)

            tm = TacticalMetric(
                analysis_id=analysis.id,
                player_id=job.player_id,
                **{k: v for k, v in tactical.items() if k != "confidence"},
            )
            session.add(tm)
        else:
            analyzer = TestAnalyzerFactory.get(test_type.value)
            calibration = None
            if calibration_config_id:
                calib = session.get(CalibrationConfig, calibration_config_id)
                if calib:
                    calibration = {
                        "meters_per_pixel": calib.meters_per_pixel,
                        "known_distances": calib.known_distances,
                        "target_positions": calib.target_positions,
                    }
            result = analyzer.analyze(
                tracks=all_tracked,
                calibration=calibration,
                video_path=str(video_path),
                fps=25.0,
            )
            analysis = Analysis(
                player_id=job.player_id,
                match_name=video.match_name,
                date=None,
                summary=f"{test_type.value} test completed via FootIQ pipeline.",
                strengths=[f"{test_type.value} analyzer confidence: {result.get('confidence', 0)}"],
                development_areas=[f"{test_type.value} analyzer calibration: {result.get('calibration_status', 'unknown')}"],
            )
            session.add(analysis)
            session.flush()
            result = analyzer.analyze(
                tracks=all_tracked,
                calibration=calibration,
                video_path=str(video_path),
                fps=25.0,
            )
            analysis = Analysis(
                player_id=job.player_id,
                match_name=video.match_name,
                date=None,
                summary=f"{test_type.value} test completed via FootIQ pipeline.",
                strengths=[f"{test_type.value} analyzer confidence: {result.get('confidence', 0)}"],
                development_areas=[f"{test_type.value} analyzer calibration: {result.get('calibration_status', 'unknown')}"],
            )
            session.add(analysis)
            session.flush()

        job.status = AnalysisStatus.completed
        job.finished_at = None
        session.add(job)
        session.commit()

        return {"status": "completed", "job_id": job_id, "analysis_id": str(analysis.id)}

    except Exception as exc:
        logger.exception("Analysis failed for job {}: {}", job_id, exc)
        if job:
            job.status = AnalysisStatus.failed
            job.error = str(exc)
            session.add(job)
            session.commit()
        return {"status": "failed", "job_id": job_id, "error": str(exc)}
