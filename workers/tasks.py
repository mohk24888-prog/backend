from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from workers.celery_app import celery_app
from db.session import get_session
from sqlalchemy import select
from db.models import (
    AnalysisJob,
    AnalysisStatus,
    Player,
    VideoUpload,
    Analysis,
    HeatmapPoint,
    MatchAction,
    PhysicalMetric,
    TacticalMetric,
)
from core.config import settings


# ---------------------------------------------------------------------------
# Optional football-player-detection-main integration
# ---------------------------------------------------------------------------
_CV_REPO_PATH = Path(__file__).resolve().parent.parent / "football-player-detection-main"
_CV_AVAILABLE = False

if _CV_REPO_PATH.exists():
    sys.path.insert(0, str(_CV_REPO_PATH))
    try:
        from src.app.app import run_video_pipeline  # noqa: E402
        from src.vision.tracker import Tracker  # noqa: E402
        from src.vision.pitch_keypoint_detector import PitchKeypointDetector  # noqa: E402
        from src.analytics.player_movement import PlayerMovementAnalyzer  # noqa: E402
        from src.core.types import Track  # noqa: E402

        _CV_AVAILABLE = True
        logger.info("Football CV repository loaded from {}", _CV_REPO_PATH)
    except Exception as exc:
        logger.warning("CV repo import failed: {}", exc)
else:
    logger.warning("CV repo not found at {}", _CV_REPO_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kmh_from_mps(mps: float) -> float:
    return mps * 3.6


def _select_subject_track(
    summaries: list[Any],
    frame_count: int,
    width: int,
    height: int,
) -> tuple[int | None, str]:
    if not summaries:
        return None, "no_tracks"
    best = None
    best_score = -1.0
    for s in summaries:
        if s.cls_name not in {"Player", "Goalkeeper"}:
            continue
        score = float(s.tracked_frames)
        score += s.frame_coverage * float(frame_count) * 0.1
        cx_score = 1.0 / (1.0 + abs(s.tracked_frames / max(frame_count, 1) - 0.5))
        score += cx_score * 10.0
        if score > best_score:
            best_score = score
            best = s
    if best is None:
        return None, "no_player_tracks"
    return int(best.track_id), "longest_track"


def _compute_individual_metrics(
    track_positions: list[tuple[int, float, float]],
    fps: float,
    pitch_length_m: float = 120.0,
    pitch_width_m: float = 70.0,
) -> dict[str, Any]:
    if len(track_positions) < 2:
        return {
            "total_distance_m": None,
            "avg_speed_kmh": None,
            "max_speed_kmh": None,
            "sprint_distance_m": None,
            "high_intensity_distance_m": None,
            "pitch_coverage_pct": None,
            "movement_time_s": None,
            "stationary_time_s": None,
            "left_side_pct": None,
            "right_side_pct": None,
            "central_zone_pct": None,
            "defensive_zone_pct": None,
            "middle_zone_pct": None,
            "attacking_zone_pct": None,
            "direction_changes": None,
            "movement_intensity": None,
            "confidence": "insufficient_data",
        }

    total_dist_m = 0.0
    max_speed_mps = 0.0
    sprint_dist_m = 0.0
    high_intensity_dist_m = 0.0
    movement_time_s = 0.0
    stationary_time_s = 0.0
    direction_changes = 0
    prev_speed_mps = 0.0
    prev_dx = 0.0
    prev_dy = 0.0

    left_frames = 0
    right_frames = 0
    central_frames = 0
    defensive_frames = 0
    middle_frames = 0
    attacking_frames = 0
    total_frames = 0

    for i in range(1, len(track_positions)):
        _, x1, y1 = track_positions[i - 1]
        _, x2, y2 = track_positions[i]
        dx = x2 - x1
        dy = y2 - y1
        dist_m = math.sqrt(dx * dx + dy * dy)
        dt_s = 1.0 / fps if fps > 0 else 0.033
        speed_mps = dist_m / dt_s if dt_s > 0 else 0.0

        total_dist_m += dist_m
        if speed_mps > max_speed_mps:
            max_speed_mps = speed_mps
        if speed_mps > 7.0:
            sprint_dist_m += dist_m
        if speed_mps > 5.5:
            high_intensity_dist_m += dist_m
        if speed_mps > 0.5:
            movement_time_s += dt_s
        else:
            stationary_time_s += dt_s

        if i > 1:
            _, px2, py2 = track_positions[i]
            _, px1, py1 = track_positions[i - 1]
            cdx = px2 - px1
            cdy = py2 - py1
            if cdx != prev_dx or cdy != prev_dy:
                if prev_dx != 0 or prev_dy != 0:
                    dot = prev_dx * cdx + prev_dy * cdy
                    dot /= (math.sqrt(prev_dx**2 + prev_dy**2) * math.sqrt(cdx**2 + cdy**2) + 1e-9)
                    if dot < 0.3:
                        direction_changes += 1
            prev_dx = cdx
            prev_dy = cdy

        norm_x = max(0.0, min(1.0, x2 / pitch_length_m))
        norm_y = max(0.0, min(1.0, y2 / pitch_width_m))
        if norm_x < 0.35:
            defensive_frames += 1
        elif norm_x > 0.65:
            attacking_frames += 1
        else:
            middle_frames += 1
        if norm_y < 0.35:
            left_frames += 1
        elif norm_y > 0.65:
            right_frames += 1
        else:
            central_frames += 1
        total_frames += 1

    total_time_s = movement_time_s + stationary_time_s
    avg_speed_kmh = _kmh_from_mps(total_dist_m / movement_time_s) if movement_time_s > 0 else 0.0
    pitch_coverage_pct = min(100.0, (len(set((x, y) for _, x, y in track_positions)) / max(len(track_positions), 1)) * 100.0)

    return {
        "total_distance_m": round(total_dist_m, 2),
        "avg_speed_kmh": round(avg_speed_kmh, 2),
        "max_speed_kmh": round(_kmh_from_mps(max_speed_mps), 2),
        "sprint_distance_m": round(sprint_dist_m, 2),
        "high_intensity_distance_m": round(high_intensity_dist_m, 2),
        "pitch_coverage_pct": round(pitch_coverage_pct, 2),
        "movement_time_s": round(movement_time_s, 2),
        "stationary_time_s": round(stationary_time_s, 2),
        "left_side_pct": round(left_frames / max(total_frames, 1) * 100, 2),
        "right_side_pct": round(right_frames / max(total_frames, 1) * 100, 2),
        "central_zone_pct": round(central_frames / max(total_frames, 1) * 100, 2),
        "defensive_zone_pct": round(defensive_frames / max(total_frames, 1) * 100, 2),
        "middle_zone_pct": round(middle_frames / max(total_frames, 1) * 100, 2),
        "attacking_zone_pct": round(attacking_frames / max(total_frames, 1) * 100, 2),
        "direction_changes": direction_changes,
        "movement_intensity": round(high_intensity_dist_m / max(total_dist_m, 0.01), 4),
        "confidence": "medium" if total_frames > 10 else "low",
    }


def _build_overlay_data(
    track_positions: list[tuple[int, float, float]],
    ball_positions: list[tuple[int, float, float]],
    events: list[dict],
    fps: float,
    width: int,
    height: int,
    pitch_length_cm: float = 12000.0,
    pitch_width_cm: float = 7000.0,
) -> dict[str, Any]:
    frames = []
    for frame_idx, frame_num, x_cm, y_cm in track_positions:
        t = frame_idx / fps if fps > 0 else frame_idx * 0.033
        px = (x_cm / pitch_length_cm) * width
        py = (y_cm / pitch_width_cm) * height
        ball_entry = next(
            (b for b in ball_positions if b[0] == frame_idx),
            None,
        )
        ball_xy = {"x": ball_entry[1], "y": ball_entry[2]} if ball_entry else None
        evt = next((e for e in events if e.get("frame_idx") == frame_idx), None)
        frames.append({
            "t": round(t, 4),
            "player": {
                "x": round(px, 2),
                "y": round(py, 2),
                "bbox": [
                    max(0, int(px - 20)),
                    max(0, int(py - 40)),
                    min(width, int(px + 20)),
                    min(height, int(py + 10)),
                ],
                "conf": 0.9,
                "track_id": frame_num,
            },
            "pitch_position": {"x": round(x_cm / 100.0, 2), "y": round(y_cm / 100.0, 2)},
            "speed_kmh": None,
            "ball": ball_xy,
            "possession": False,
            "event": evt.get("type") if evt else None,
        })
    for i in range(1, len(frames)):
        dx = frames[i]["player"]["x"] - frames[i - 1]["player"]["x"]
        dy = frames[i]["player"]["y"] - frames[i - 1]["player"]["y"]
        dist_px = math.sqrt(dx * dx + dy * dy)
        dt_s = 1.0 / fps if fps > 0 else 0.033
        speed_kmh = (dist_px * 0.036 / dt_s) if dt_s > 0 else 0.0
        frames[i]["speed_kmh"] = round(min(speed_kmh, 40.0), 2)
    return {"frames": frames}


def _build_overlay_data_multiplayer(
    all_tracks_by_frame: dict[int, list[dict]],
    ball_positions: list[tuple[int, float, float]],
    events: list[dict],
    fps: float,
    width: int,
    height: int,
    subject_track_id: int,
    pitch_length_cm: float = 12000.0,
    pitch_width_cm: float = 7000.0,
) -> dict[str, Any]:
    frames = []
    sorted_frames = sorted(all_tracks_by_frame.keys())
    for frame_idx in sorted_frames:
        tracks = all_tracks_by_frame[frame_idx]
        t = frame_idx / fps if fps > 0 else frame_idx * 0.033
        
        ball_entry = next((b for b in ball_positions if b[0] == frame_idx), None)
        ball_xy = {"x": ball_entry[1], "y": ball_entry[2]} if ball_entry else None
        
        players = []
        for trk in tracks:
            px_m = trk.get("pitch_x_m", 0.0)
            py_m = trk.get("pitch_y_m", 0.0)
            px = (px_m / (pitch_length_cm / 100.0)) * width
            py = (py_m / (pitch_width_cm / 100.0)) * height
            team_id = trk.get("team_id")
            is_subject = trk.get("track_id") == subject_track_id
            
            players.append({
                "track_id": trk.get("track_id"),
                "cls_name": trk.get("cls_name", "Player"),
                "team_id": team_id,
                "team_conf": trk.get("team_conf"),
                "conf": trk.get("conf", 0.0),
                "x": round(px, 2),
                "y": round(py, 2),
                "bbox": [
                    max(0, int(px - 20)),
                    max(0, int(py - 40)),
                    min(width, int(px + 20)),
                    min(height, int(py + 10)),
                ] if trk.get("x1") is None else [
                    int(trk["x1"]),
                    int(trk["y1"]),
                    int(trk["x2"]),
                    int(trk["y2"]),
                ],
                "pitch_position": {"x": round(px_m, 2), "y": round(py_m, 2)},
                "is_subject": is_subject,
            })
        
        evt = next((e for e in events if e.get("frame_idx") == frame_idx), None)
        frames.append({
            "t": round(t, 4),
            "players": players,
            "ball": ball_xy,
            "possession": False,
            "event": evt.get("type") if evt else None,
        })
    return {"frames": frames}


def _build_heatmap_points(
    track_positions: list[tuple[int, float, float]],
    pitch_length_cm: float = 12000.0,
    pitch_width_cm: float = 7000.0,
    grid_x: int = 12,
    grid_y: int = 8,
) -> list[dict[str, Any]]:
    grid = {}
    for _, x_cm, y_cm in track_positions:
        gx = int((x_cm / pitch_length_cm) * grid_x)
        gy = int((y_cm / pitch_width_cm) * grid_y)
        gx = max(0, min(grid_x - 1, gx))
        gy = max(0, min(grid_y - 1, gy))
        grid[(gx, gy)] = grid.get((gx, gy), 0) + 1
    max_count = max(grid.values()) if grid else 1
    points = []
    for (gx, gy), count in grid.items():
        points.append({
            "x": gx / max(grid_x - 1, 1),
            "y": gy / max(grid_y - 1, 1),
            "intensity": round(count / max_count, 4),
        })
    return points


def _build_trajectory_points(
    track_positions: list[tuple[int, float, float]],
    pitch_length_cm: float = 12000.0,
    pitch_width_cm: float = 7000.0,
    sample_every: int = 5,
) -> list[dict[str, Any]]:
    pts = []
    for idx, (frame_idx, x_cm, y_cm) in enumerate(track_positions):
        if idx % sample_every != 0:
            continue
        pts.append({
            "frame": frame_idx,
            "x": round(x_cm / 100.0, 2),
            "y": round(y_cm / 100.0, 2),
        })
    return pts


def _build_events(
    track_positions: list[tuple[int, float, float]],
    speed_series: list[tuple[int, float]],
    ball_positions: list[tuple[int, float, float]],
    fps: float,
) -> list[dict[str, Any]]:
    events = []
    prev_speed = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for i, (frame_idx, x_cm, y_cm) in enumerate(track_positions):
        speed = next((s for fr, s in speed_series if fr == frame_idx), 0.0)
        if speed > 25.0:
            events.append({
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 2) if fps > 0 else 0,
                "type": "sprint",
                "label": "Sprint",
                "severity": "high",
            })
        elif speed > 20.0:
            events.append({
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 2) if fps > 0 else 0,
                "type": "high_speed_run",
                "label": "High Speed Run",
                "severity": "medium",
            })
        if i > 0:
            dx = x_cm - prev_x
            dy = y_cm - prev_y
            dot = (prev_x * dx + prev_y * dy) / (math.sqrt(prev_x**2 + prev_y**2) * math.sqrt(dx**2 + dy**2) + 1e-9)
            if dot < -0.5 and speed > 3.0:
                events.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 2) if fps > 0 else 0,
                    "type": "direction_change",
                    "label": "Direction Change",
                    "severity": "low",
                })
        prev_speed = speed
        prev_x = x_cm
        prev_y = y_cm
    for frame_idx, bx, by in ball_positions:
        if not any(e["frame_idx"] == frame_idx and e["type"] == "ball_possession" for e in events):
            events.append({
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 2) if fps > 0 else 0,
                "type": "ball_possession",
                "label": "Ball Possession",
                "severity": "medium",
            })
    events.sort(key=lambda e: e["frame_idx"])
    return events


def _build_simulation(
    subject_track_id: int,
    subject_positions: list[tuple[int, float, float]],
    pitch_length_cm: float = 12000.0,
    pitch_width_cm: float = 7000.0,
    formation: str = "4-3-3",
    position: str = "CM",
) -> dict[str, Any]:
    FORMATIONS = {
        "4-3-3": [
            ("GK", 100.0, 50.0),
            ("LB", 75.0, 15.0),
            ("CB", 75.0, 35.0),
            ("CB", 75.0, 65.0),
            ("RB", 75.0, 85.0),
            ("CM", 55.0, 30.0),
            ("CM", 55.0, 50.0),
            ("CM", 55.0, 70.0),
            ("LW", 35.0, 20.0),
            ("RW", 35.0, 80.0),
            ("ST", 25.0, 50.0),
        ],
        "4-4-2": [
            ("GK", 100.0, 50.0),
            ("LB", 75.0, 15.0),
            ("CB", 75.0, 35.0),
            ("CB", 75.0, 65.0),
            ("RB", 75.0, 85.0),
            ("LM", 55.0, 15.0),
            ("CM", 55.0, 35.0),
            ("CM", 55.0, 65.0),
            ("RM", 55.0, 85.0),
            ("ST", 30.0, 40.0),
            ("ST", 30.0, 60.0),
        ],
        "3-5-2": [
            ("GK", 100.0, 50.0),
            ("CB", 75.0, 25.0),
            ("CB", 75.0, 50.0),
            ("CB", 75.0, 75.0),
            ("LWB", 60.0, 10.0),
            ("CM", 55.0, 35.0),
            ("CM", 55.0, 65.0),
            ("RWB", 60.0, 90.0),
            ("ST", 30.0, 40.0),
            ("ST", 30.0, 60.0),
        ],
        "4-2-3-1": [
            ("GK", 100.0, 50.0),
            ("LB", 75.0, 15.0),
            ("CB", 75.0, 35.0),
            ("CB", 75.0, 65.0),
            ("RB", 75.0, 85.0),
            ("DM", 65.0, 35.0),
            ("DM", 65.0, 65.0),
            ("AM", 45.0, 30.0),
            ("AM", 45.0, 50.0),
            ("AM", 45.0, 70.0),
            ("ST", 25.0, 50.0),
        ],
    }

    positions = FORMATIONS.get(formation, FORMATIONS["4-3-3"])
    subject_idx = None
    for idx, (role, _, _) in enumerate(positions):
        if role == position:
            subject_idx = idx
            break
    if subject_idx is None:
        subject_idx = 6 if len(positions) > 6 else len(positions) // 2

    teammates = []
    opponents = []
    for idx, (role, base_x_pct, base_y_pct) in enumerate(positions):
        if idx == subject_idx:
            continue
        x_cm = (base_x_pct / 100.0) * pitch_length_cm
        y_cm = (base_y_pct / 100.0) * pitch_width_cm
        entry = {
            "role": role,
            "base_position_cm": {"x": round(x_cm, 1), "y": round(y_cm, 1)},
            "movement_zone_cm": {
                "x_min": round(max(0, x_cm - 500), 1),
                "x_max": round(min(pitch_length_cm, x_cm + 500), 1),
                "y_min": round(max(0, y_cm - 400), 1),
                "y_max": round(min(pitch_width_cm, y_cm + 400), 1),
            },
            "observed": False,
            "simulated": True,
        }
        if idx < 6:
            teammates.append(entry)
        else:
            opponents.append(entry)

    subject_avg_x = (
        sum(x for _, x, y in subject_positions) / len(subject_positions)
        if subject_positions
        else pitch_length_cm * 0.5
    )
    interpretation = (
        f"Based on {len(subject_positions)} tracked positions, "
        f"the player's average position is at {subject_avg_x / 100:.1f}m from the defensive line. "
        f"As a {position}, movement patterns suggest "
        f"{'high attacking presence' if subject_avg_x < pitch_length_cm * 0.4 else 'balanced box-to-box coverage' if subject_avg_x < pitch_length_cm * 0.6 else 'defensive solidity and build-up role'}."
    )

    return {
        "subject_position": position,
        "formation": formation,
        "subject_track_id": subject_track_id,
        "teammates": teammates,
        "opponents": opponents,
        "tactical_interpretation": interpretation,
        "labels": {
            "observed": "OBSERVED",
            "simulated": "SIMULATED",
        },
    }


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------


@celery_app.task(name="workers.tasks.analyze_video", bind=True)
def analyze_video(self, job_id: str) -> dict:
    import time
    _task_start = time.time()
    session_gen = get_session()
    session = next(session_gen)
    job = None

    try:
        job = session.get(AnalysisJob, job_id)
        if not job:
            logger.error("Analysis job not found: {}", job_id)
            return {"status": "failed", "error": "job_not_found"}

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

        job.status = AnalysisStatus.preprocessing
        job.worker = "cpu"
        session.add(job)
        session.commit()
        logger.info("Analyzing video: {}", video_path)

        # ---------------------------------------------------------------
        # Run CV pipeline
        # ---------------------------------------------------------------
        warnings: list[str] = []
        subject_track_id: int | None = None
        selection_method = "none"
        overlay_data: dict[str, Any] = {"frames": []}
        metrics: dict[str, Any] = {}
        heatmap_points: list[dict[str, Any]] = []
        trajectory_points: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        simulation: dict[str, Any] = {}
        processed_frames = 0
        fps = 25.0
        vid_width = 1280
        vid_height = 720

        if _CV_AVAILABLE:
            try:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(str(video_path))
                if _cap.isOpened():
                    _w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                    _h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                    _f = float(_cap.get(_cv2.CAP_PROP_FPS))
                    if _w > 0:
                        vid_width = _w
                    if _h > 0:
                        vid_height = _h
                    if _f > 0:
                        fps = _f
                _cap.release()

                job.status = AnalysisStatus.detecting
                session.add(job)
                session.commit()

                run_output_dir = Path(settings.processed_dir) / f"job_{job_id}"
                run_output_dir.mkdir(parents=True, exist_ok=True)

                tracker_weights = _CV_REPO_PATH / "models" / "best_players_gk_1280_s_e300.pt"
                ball_weights = _CV_REPO_PATH / "models" / "ball_tracking_1280_e300.pt"
                pitch_weights = _CV_REPO_PATH / "models" / "pitch_kpts32_y8s_640_e500_AO.pt"
                tracker_cfg = _CV_REPO_PATH / "configs" / "botsort.yaml"

                for p in [tracker_weights, ball_weights, pitch_weights, tracker_cfg]:
                    if not p.exists():
                        warnings.append(f"Model file not found: {p}")

                tracker = Tracker(
                    weights=str(tracker_weights),
                    device="cpu",
                    image_size=1280,
                    tracker_cfg=str(tracker_cfg),
                )
                pitch_detector = PitchKeypointDetector(
                    weights=str(pitch_weights),
                    device="cpu",
                    image_size=640,
                    num_keypoints=32,
                )

                def ball_overlay_fn(frame):
                    from src.vision.ball import BallAnnotator, BallTracker
                    from src.core.types import Detection, Track as TrackCls

                    model = __import__("ultralytics").YOLO(str(ball_weights)).to("cpu")
                    ball_tracker = BallTracker(buffer_size=20)
                    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

                    def fallback(image_slice):
                        result = model(image_slice, imgsz=640, verbose=False)[0]
                        return __import__("supervision").Detections.from_ultralytics(result)

                    slicer = __import__("supervision").InferenceSlicer(
                        callback=fallback,
                        slice_wh=(640, 640),
                        overlap_filter_strategy=__import__("supervision").OverlapFilter.NONE,
                    )
                    detections = slicer(frame).with_nms(threshold=0.1)
                    detections = ball_tracker.update(detections)
                    annotated = ball_annotator.annotate(frame.copy(), detections)
                    ball_xy = None
                    if len(detections) > 0:
                        xy = detections.get_anchors_coordinates(__import__("supervision").Position.BOTTOM_CENTER)
                        ball_xy = (float(xy[0][0]), float(xy[0][1]))
                    return annotated, ball_xy

                job.status = AnalysisStatus.tracking
                session.add(job)
                session.commit()

                processed_frames = run_video_pipeline(
                    video_path=video_path,
                    tracker=tracker,
                    pitch_detector=pitch_detector,
                    yolo_video_path=run_output_dir / "yolo_tracking.mp4",
                    team_homography_video_path=run_output_dir / "tactical_view.mp4",
                    vid_stride=settings.frame_skip,
                    enable_team_assignment=True,
                    ball_overlay_fn=ball_overlay_fn,
                    enable_possession=True,
                    heatmap_output_dir=run_output_dir / "heatmaps",
                    heatmap_top_n=0,
                    heatmap_track_ids=None,
                    trajectory_output_dir=run_output_dir / "trajectories",
                    trajectory_views=None,
                    csv_output_path=run_output_dir / "tracks.csv",
                )

                job.status = AnalysisStatus.calculating
                session.add(job)
                session.commit()

                csv_path = run_output_dir / "tracks.csv"
                if csv_path.exists():
                    import csv as csv_mod
                    track_positions = []
                    ball_positions = []
                    all_tracks_by_frame = {}
                    with open(csv_path, newline="", encoding="utf-8") as f:
                        reader = csv_mod.DictReader(f)
                        for row in reader:
                            frame_idx = int(row.get("frame", row.get("frame_idx", 0)))
                            cls_name = row.get("cls_name", "")
                            team_id = row.get("team_id", "-1")
                            team_conf = row.get("team_conf", "-1")
                            conf = row.get("conf", "0")
                            x1 = row.get("x1")
                            y1 = row.get("y1")
                            x2 = row.get("x2")
                            y2 = row.get("y2")
                            cx = row.get("cx")
                            cy = row.get("cy")
                            
                            if cls_name == "Ball":
                                bx = row.get("pitch_x_m")
                                by = row.get("pitch_y_m")
                                if bx and by:
                                    try:
                                        ball_positions.append((frame_idx, float(bx), float(by)))
                                    except ValueError:
                                        pass
                            elif cls_name in {"Player", "Goalkeeper"}:
                                px = row.get("pitch_x_m")
                                py = row.get("pitch_y_m")
                                tid = row.get("track_id")
                                if px and py and tid:
                                    try:
                                        track_positions.append((int(tid), float(px), float(py)))
                                        if frame_idx not in all_tracks_by_frame:
                                            all_tracks_by_frame[frame_idx] = []
                                        all_tracks_by_frame[frame_idx].append({
                                            "track_id": int(tid),
                                            "cls_name": cls_name,
                                            "team_id": int(team_id) if team_id != "-1" else None,
                                            "team_conf": float(team_conf) if team_conf != "-1" else None,
                                            "conf": float(conf),
                                            "x1": float(x1) if x1 else None,
                                            "y1": float(y1) if y1 else None,
                                            "x2": float(x2) if x2 else None,
                                            "y2": float(y2) if y2 else None,
                                            "cx": float(cx) if cx else None,
                                            "cy": float(cy) if cy else None,
                                            "pitch_x_m": float(px),
                                            "pitch_y_m": float(py),
                                        })
                                    except ValueError:
                                        pass

                    subject_track_id, selection_method = _select_subject_track(
                        [type("S", (), {"track_id": tid, "cls_name": "Player", "tracked_frames": sum(1 for t, _, _ in track_positions if t == tid), "frame_coverage": 0.5}) for tid in {t for t, _, _ in track_positions}],
                        processed_frames,
                        vid_width,
                        vid_height,
                    )
                    if subject_track_id is not None:
                        subject_positions = [(f, x, y) for tid, x, y in track_positions if tid == subject_track_id]
                        metrics = _compute_individual_metrics(subject_positions, fps)
                        heatmap_points = _build_heatmap_points(subject_positions)
                        trajectory_points = _build_trajectory_points(subject_positions)
                        speed_series = [(f, metrics.get("avg_speed_kmh", 0.0) / 3.6) for f, _, _ in subject_positions]
                        events = _build_events(subject_positions, speed_series, ball_positions, fps)
                        simulation = _build_simulation(
                            subject_track_id,
                            subject_positions,
                            formation="4-3-3",
                            position="CM",
                        )
                        overlay_data = _build_overlay_data_multiplayer(
                            all_tracks_by_frame,
                            ball_positions,
                            events,
                            fps,
                            vid_width,
                            vid_height,
                            subject_track_id=subject_track_id,
                        )
                else:
                    warnings.append("tracks_csv_missing")

                job.status = AnalysisStatus.generating_report
                session.add(job)
                session.commit()

            except Exception as exc:
                logger.exception("CV pipeline failed: {}", exc)
                warnings.append(f"cv_pipeline_error: {exc}")
                job.status = AnalysisStatus.failed
                job.error = str(exc)
                session.add(job)
                session.commit()
                return {"status": "failed", "job_id": job_id, "error": str(exc)}

        # ---------------------------------------------------------------
        # Fallback if CV repo unavailable or failed
        # ---------------------------------------------------------------
        if not overlay_data.get("frames") and not metrics:
            warnings.append("cv_repo_unavailable_used_fallback")
            from services.analysis.analyzers.factory import TestAnalyzerFactory

            all_tracked = []
            job.status = AnalysisStatus.detecting
            session.add(job)
            session.commit()
            try:
                import cv2
                import numpy as np
                from ultralytics import YOLO

                model = YOLO(settings.yolo_model)
                cap = cv2.VideoCapture(str(video_path))
                frame_id = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_id % settings.frame_skip != 0:
                        frame_id += 1
                        continue
                    results = model(frame, conf=settings.confidence_threshold, device="cpu", verbose=False)
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            all_tracked.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": float(box.conf[0]),
                                "class_id": int(box.cls[0]),
                                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                                "frame_id": frame_id,
                                "track_id": 0,
                            })
                    frame_id += 1
                cap.release()
            except Exception:
                all_tracked = []

            physical = MetricsCalculator.physical_metrics(all_tracked)
            tactical = MetricsCalculator.tactical_metrics(all_tracked)
            heatmap_points = MetricsCalculator.heatmap(all_tracked)
            metrics = {
                "total_distance_m": physical.get("total_distance_m"),
                "avg_speed_kmh": round(physical.get("avg_speed_ms", 0) * 3.6, 2),
                "max_speed_kmh": None,
                "sprint_distance_m": None,
                "high_intensity_distance_m": None,
                "pitch_coverage_pct": None,
                "movement_time_s": None,
                "stationary_time_s": None,
                "left_side_pct": None,
                "right_side_pct": None,
                "central_zone_pct": None,
                "defensive_zone_pct": None,
                "middle_zone_pct": None,
                "attacking_zone_pct": None,
                "direction_changes": None,
                "movement_intensity": None,
                "confidence": physical.get("confidence", "low"),
            }

        # ---------------------------------------------------------------
        # Persist results
        # ---------------------------------------------------------------
        job.status = AnalysisStatus.generating_report
        session.add(job)
        session.commit()

        _max_speed_kmh = metrics.get("max_speed_kmh")
        _avg_speed_kmh = metrics.get("avg_speed_kmh")
        _max_speed_ms = (_max_speed_kmh / 3.6) if _max_speed_kmh is not None else None
        _avg_speed_ms = (_avg_speed_kmh / 3.6) if _avg_speed_kmh is not None else None

        analysis = Analysis(
            player_id=job.player_id,
            job_id=job.id,
            match_name=video.match_name,
            date=None,
            overall_rating=None,
            technical=None,
            tactical=None,
            physical=None,
            mental=None,
            summary="Individual player analysis completed via FootIQ CV pipeline." if _CV_AVAILABLE else "Analysis completed via FootIQ fallback pipeline.",
            strengths=["Player movement detected and tracked"] if metrics.get("total_distance_m") else ["Limited data available"],
            development_areas=["Ball detection limited — solo video without clear ball"] if not ball_positions else ["Continue training for improved consistency"],
            cv_repo_used="football-player-detection-main" if _CV_AVAILABLE else "fallback",
            subject_track_id=subject_track_id,
            selection_method=selection_method,
            pipeline_warnings=warnings if warnings else None,
            overlay_data=overlay_data if overlay_data.get("frames") else None,
            simulation_data=simulation if simulation else None,
            video_url=video.public_url,
            analysis_duration_s=round(time.time() - _task_start, 2),
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

        for evt in events:
            action = MatchAction(
                analysis_id=analysis.id,
                player_id=job.player_id,
                minute=int(evt.get("timestamp_s", 0) / 60),
                type=evt.get("type", "movement"),
                description=evt.get("label", ""),
                category="Individual",
                success=None,
            )
            session.add(action)

        if metrics.get("total_distance_m") is not None:
            pm = PhysicalMetric(
                analysis_id=analysis.id,
                player_id=job.player_id,
                total_distance_m=metrics.get("total_distance_m"),
                max_speed_ms=_max_speed_ms,
                avg_speed_ms=_avg_speed_ms,
                sprint_count=None,
                hi_run_count=None,
                acceleration_profile=metrics,
                speed_zones=None,
                fatigue_index=None,
            )
            session.add(pm)

        job.status = AnalysisStatus.completed
        job.finished_at = None
        session.add(job)
        session.commit()

        logger.info(
            "Analysis complete: job={}, analysis={}, subject_track={}, frames={}",
            job_id,
            analysis.id,
            subject_track_id,
            len(overlay_data.get("frames", [])),
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "analysis_id": str(analysis.id),
            "subject_track_id": subject_track_id,
            "cv_repo": "football-player-detection-main" if _CV_AVAILABLE else "fallback",
            "frames_sampled": len(overlay_data.get("frames", [])),
            "warnings": warnings,
        }

    except Exception as exc:
        logger.exception("Analysis failed for job {}: {}", job_id, exc)
        if job:
            job.status = AnalysisStatus.failed
            job.error = str(exc)
            session.add(job)
            session.commit()
        return {"status": "failed", "job_id": job_id, "error": str(exc)}
