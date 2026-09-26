"""
CV integration service for FootIQ.

Wraps ``football-player-detection-main`` pipeline for individual player analysis.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from ultralytics import YOLO

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CV repo path
# ---------------------------------------------------------------------------
CV_REPO_DIR = Path(
    r"C:\Users\mohamed\Downloads\gfn\football-player-detection-main"
)
CV_SRC_DIR = CV_REPO_DIR / "src"
CV_MODELS_DIR = CV_REPO_DIR / "models"
CV_CONFIGS_DIR = CV_REPO_DIR / "configs"

if str(CV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CV_SRC_DIR))

# ---------------------------------------------------------------------------
# Import CV modules (fail fast if unavailable)
# ---------------------------------------------------------------------------
try:
    from src.app.app import run_video_pipeline  # noqa: E402
    from src.core.pitch import SoccerPitchConfiguration  # noqa: E402
    from src.core.types import Track as CVTrack  # noqa: E402
    from src.vision.pitch_keypoint_detector import (  # noqa: E402
        PitchKeypointDetector,
    )
    from src.vision.tracker import Tracker as CVTracker  # noqa: E402
    from src.vision.ball import BallAnnotator, BallTracker  # noqa: E402
    import supervision as sv  # noqa: E402

    CV_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        f"football-player-detection-main is required but could not be imported: {exc}. "
        "Ensure the repository is present at the expected path and its dependencies are installed."
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(preferred: str = "auto") -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Ball overlay builder (mirrors main.py build_ball_overlay_fn)
# ---------------------------------------------------------------------------

def _build_ball_overlay_fn(
    ball_weights: Path,
    device: str,
) -> Any:
    model = YOLO(str(ball_weights)).to(device)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def _fallback(image_slice: np.ndarray) -> "sv.Detections":
        result = model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=_fallback,
        slice_wh=(640, 640),
        overlap_filter=sv.OverlapFilter.NONE,
    )

    def overlay_fn(
        frame: np.ndarray,
    ) -> tuple[np.ndarray, Optional[tuple[float, float]]]:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated = ball_annotator.annotate(frame.copy(), detections)
        if len(detections) == 0:
            return annotated, None
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        return annotated, (float(xy[0][0]), float(xy[0][1]))

    return overlay_fn


# ---------------------------------------------------------------------------
# Subject identification from CSV
# ---------------------------------------------------------------------------

def _read_tracks_csv(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _identify_subject_track_id(
    rows: list[dict[str, str]],
) -> tuple[int, str]:
    """Return (track_id, selection_method) for the subject player."""
    frame_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        tid = int(row["track_id"])
        cls = row.get("cls_name", "")
        if cls in {"Player", "Goalkeeper"} and tid >= 0:
            frame_counts[tid] += 1

    if not frame_counts:
        raise ValueError("No clear player detected")

    subject_tid = max(frame_counts, key=frame_counts.get)  # type: ignore[arg-type]
    return int(subject_tid), "longest_track"


# ---------------------------------------------------------------------------
# Metrics computation from CSV rows
# ---------------------------------------------------------------------------

def _compute_subject_metrics(
    rows: list[dict[str, str]],
    subject_tid: int,
    fps: float,
    pitch_cfg: SoccerPitchConfiguration,
) -> dict[str, Any]:
    """Compute per-player metrics from CSV rows."""
    subject_rows = [
        r for r in rows
        if int(r["track_id"]) == subject_tid and r.get("pitch_point_valid") == "1"
    ]
    subject_rows.sort(key=lambda r: int(r["frame"]))

    if len(subject_rows) < 2:
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
            "central_pct": None,
            "defensive_zone_pct": None,
            "middle_zone_pct": None,
            "attacking_zone_pct": None,
            "tracked_frames": len(subject_rows),
            "confidence": "low",
        }

    pitch_length_cm = float(pitch_cfg.length)
    pitch_width_cm = float(pitch_cfg.width)

    distances_m: list[float] = []
    speeds_kmh: list[float] = []
    sprint_dist_m: float = 0.0
    high_intensity_dist_m: float = 0.0
    movement_time_s: float = 0.0
    stationary_time_s: float = 0.0

    left_frames = 0
    right_frames = 0
    central_frames = 0
    defensive_frames = 0
    middle_frames = 0
    attacking_frames = 0

    prev_x = prev_y = None
    prev_frame = None

    for row in subject_rows:
        x_cm = float(row["pitch_x_m"])
        y_cm = float(row["pitch_y_m"])
        frame = int(row["frame"])

        if y_cm < pitch_width_cm / 3:
            left_frames += 1
        elif y_cm < 2 * pitch_width_cm / 3:
            central_frames += 1
        else:
            right_frames += 1

        if x_cm < pitch_length_cm / 3:
            defensive_frames += 1
        elif x_cm < 2 * pitch_length_cm / 3:
            middle_frames += 1
        else:
            attacking_frames += 1

        if (
            prev_x is not None
            and prev_y is not None
            and prev_frame is not None
        ):
            frame_diff = frame - prev_frame
            if frame_diff > 0 and fps > 0:
                dt_s = frame_diff / fps
                dist_cm = math.hypot(x_cm - prev_x, y_cm - prev_y)
                dist_m = dist_cm / 100.0
                distances_m.append(dist_m)
                if dt_s > 0:
                    speed_mps = dist_m / dt_s
                    speed_kmh_val = speed_mps * 3.6
                    speeds_kmh.append(speed_kmh_val)
                    if speed_kmh_val > 25.0:
                        sprint_dist_m += dist_m
                    if speed_kmh_val > 20.0:
                        high_intensity_dist_m += dist_m
                    if speed_kmh_val > 0.5:
                        movement_time_s += dt_s
                    else:
                        stationary_time_s += dt_s

        prev_x, prev_y, prev_frame = x_cm, y_cm, frame

    total_frames = len(subject_rows)
    total_distance_m = round(sum(distances_m), 2) if distances_m else None
    avg_speed_kmh = round(sum(speeds_kmh) / len(speeds_kmh), 2) if speeds_kmh else None
    max_speed_kmh = round(max(speeds_kmh), 2) if speeds_kmh else None
    sprint_distance_m = round(sprint_dist_m, 2) if sprint_dist_m > 0 else None
    high_intensity_distance_m = (
        round(high_intensity_dist_m, 2) if high_intensity_dist_m > 0 else None
    )

    grid_cells: set[tuple[int, int]] = set()
    for row in subject_rows:
        x_cm = float(row["pitch_x_m"])
        y_cm = float(row["pitch_y_m"])
        gx = int(x_cm // 1000)
        gy = int(y_cm // 1000)
        grid_cells.add((gx, gy))
    total_cells = int(pitch_length_cm / 1000) * int(pitch_width_cm / 1000)
    pitch_coverage_pct = (
        round(len(grid_cells) / total_cells * 100, 2) if total_cells > 0 else None
    )

    movement_time_s = round(movement_time_s, 2) if movement_time_s > 0 else None
    stationary_time_s = round(stationary_time_s, 2) if stationary_time_s > 0 else None

    total_zone = left_frames + central_frames + right_frames
    left_side_pct = round(left_frames / total_zone * 100, 2) if total_zone > 0 else None
    right_side_pct = round(right_frames / total_zone * 100, 2) if total_zone > 0 else None
    central_pct = round(central_frames / total_zone * 100, 2) if total_zone > 0 else None

    total_zone2 = defensive_frames + middle_frames + attacking_frames
    defensive_zone_pct = (
        round(defensive_frames / total_zone2 * 100, 2) if total_zone2 > 0 else None
    )
    middle_zone_pct = (
        round(middle_frames / total_zone2 * 100, 2) if total_zone2 > 0 else None
    )
    attacking_zone_pct = (
        round(attacking_frames / total_zone2 * 100, 2) if total_zone2 > 0 else None
    )

    confidence = "high" if total_frames > 100 else ("medium" if total_frames > 30 else "low")

    return {
        "total_distance_m": total_distance_m,
        "avg_speed_kmh": avg_speed_kmh,
        "max_speed_kmh": max_speed_kmh,
        "sprint_distance_m": sprint_distance_m,
        "high_intensity_distance_m": high_intensity_distance_m,
        "pitch_coverage_pct": pitch_coverage_pct,
        "movement_time_s": movement_time_s,
        "stationary_time_s": stationary_time_s,
        "left_side_pct": left_side_pct,
        "right_side_pct": right_side_pct,
        "central_pct": central_pct,
        "defensive_zone_pct": defensive_zone_pct,
        "middle_zone_pct": middle_zone_pct,
        "attacking_zone_pct": attacking_zone_pct,
        "tracked_frames": total_frames,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def _detect_events(
    subject_rows: list[dict[str, str]],
    fps: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if len(subject_rows) < 3:
        return events

    # Compute per-frame speeds
    pos_frames: list[tuple[int, float, float]] = []
    for r in subject_rows:
        pos_frames.append((
            int(r["frame"]),
            float(r["pitch_x_m"]),
            float(r["pitch_y_m"]),
        ))

    speeds: list[tuple[int, float]] = []
    for i in range(1, len(pos_frames)):
        frame = pos_frames[i][0]
        dx = pos_frames[i][1] - pos_frames[i - 1][1]
        dy = pos_frames[i][2] - pos_frames[i - 1][2]
        dist_cm = math.hypot(dx, dy)
        frame_diff = frame - pos_frames[i - 1][0]
        dt_s = (frame_diff / fps) if fps > 0 and frame_diff > 0 else 0.04
        dist_m = dist_cm / 100.0
        speed_kmh = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
        speeds.append((frame, speed_kmh))

    # High-speed runs (>= 25 km/h for >= 3 consecutive frames)
    run_start = None
    for i, (frame, spd) in enumerate(speeds):
        if spd >= 25.0:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= 3:
                events.append({
                    "type": "high_speed_run",
                    "start_frame": speeds[run_start][0],
                    "end_frame": speeds[i - 1][0],
                    "peak_speed_kmh": round(max(v for _, v in speeds[run_start:i]), 2),
                })
            run_start = None
    if run_start is not None and (len(speeds) - run_start) >= 3:
        events.append({
            "type": "high_speed_run",
            "start_frame": speeds[run_start][0],
            "end_frame": speeds[-1][0],
            "peak_speed_kmh": round(max(v for _, v in speeds[run_start:]), 2),
        })

    # Direction changes (heading change > 90 degrees over 5 frames)
    headings: list[tuple[int, float]] = []
    for i in range(1, len(pos_frames)):
        dx = pos_frames[i][1] - pos_frames[i - 1][1]
        dy = pos_frames[i][2] - pos_frames[i - 1][2]
        if math.hypot(dx, dy) > 50:
            headings.append((pos_frames[i][0], math.degrees(math.atan2(dy, dx))))

    for i in range(5, len(headings)):
        h1 = headings[i - 5][1]
        h2 = headings[i][1]
        delta = abs(h2 - h1)
        if delta > 180:
            delta = 360 - delta
        if delta > 90:
            events.append({
                "type": "direction_change",
                "frame": headings[i][0],
                "angle_change_deg": round(delta, 1),
            })
            break

    return events


# ---------------------------------------------------------------------------
# 11v11 tactical simulation
# ---------------------------------------------------------------------------

def _detect_role(avg_x_cm: float, avg_y_cm: float, pitch_cfg: SoccerPitchConfiguration) -> str:
    pitch_length = float(pitch_cfg.length)
    pitch_width = float(pitch_cfg.width)
    x_norm = avg_x_cm / pitch_length
    y_norm = avg_y_cm / pitch_width

    if x_norm < 0.25:
        if y_norm < 0.3:
            return "LB"
        if y_norm > 0.7:
            return "RB"
        return "CB"
    if x_norm < 0.5:
        if y_norm < 0.3:
            return "LM"
        if y_norm > 0.7:
            return "RM"
        return "CM"
    if x_norm < 0.75:
        if y_norm < 0.3:
            return "LW"
        if y_norm > 0.7:
            return "RW"
        return "CM"
    if y_norm < 0.35:
        return "LW"
    if y_norm > 0.65:
        return "RW"
    return "ST"


def _generate_simulation(
    subject_tid: int,
    subject_avg_x_cm: float,
    subject_avg_y_cm: float,
    subject_rows: list[dict[str, str]],
    pitch_cfg: SoccerPitchConfiguration,
    fps: float,
) -> dict[str, Any]:
    pitch_length = float(pitch_cfg.length)
    pitch_width = float(pitch_cfg.width)

    role = _detect_role(subject_avg_x_cm, subject_avg_y_cm, pitch_cfg)

    used_positions: set[tuple[int, int]] = set()
    subject_pos = (int(round(subject_avg_x_cm)), int(round(subject_avg_y_cm)))
    used_positions.add(subject_pos)

    def _place_players(
        base_positions: list[tuple[str, tuple[float, float]]], flip_x: bool
    ) -> list[dict[str, Any]]:
        players = []
        for role_name, (bx, by) in base_positions:
            px = int(round(pitch_length - bx)) if flip_x else int(round(bx))
            py = int(round(by))
            gx = int(round(px / 500.0)) * 500
            gy = int(round(py / 500.0)) * 500
            gx = max(500, min(int(pitch_length) - 500, gx))
            gy = max(500, min(int(pitch_width) - 500, gy))
            key = (gx, gy)
            attempts = 0
            while key in used_positions and attempts < 20:
                gx += 250 if attempts % 2 == 0 else -250
                gy += 250 if attempts % 3 == 0 else -250
                gx = max(500, min(int(pitch_length) - 500, gx))
                gy = max(500, min(int(pitch_width) - 500, gy))
                key = (gx, gy)
                attempts += 1
            used_positions.add(key)
            players.append({
                "role": role_name,
                "position": {"x": gx, "y": gy},
                "movement_zone": {
                    "x_min": max(0, gx - 1500),
                    "x_max": min(int(pitch_length), gx + 1500),
                    "y_min": max(0, gy - 1000),
                    "y_max": min(int(pitch_width), gy + 1000),
                },
            })
        return players

    teammate_formation = [
        ("GK", (500, 3500)),
        ("CB", (2200, 2500)),
        ("CB", (2200, 4500)),
        ("LB", (2200, 800)),
        ("RB", (2200, 6200)),
        ("CM", (5500, 2500)),
        ("CM", (5500, 4500)),
        ("LW", (9000, 1000)),
        ("ST", (9000, 3500)),
        ("RW", (9000, 6000)),
    ]
    teammate_formation = [
        p for p in teammate_formation
        if not (p[0] == "CM" and p[1] == (5500, 3500))
    ]
    teammates_raw = _place_players(teammate_formation, flip_x=False)
    teammates = [
        {
            "role": role,
            "position": {"x": subject_pos[0], "y": subject_pos[1]},
            "movement_zone": {
                "x_min": max(0, subject_pos[0] - 1500),
                "x_max": min(int(pitch_length), subject_pos[0] + 1500),
                "y_min": max(0, subject_pos[1] - 1000),
                "y_max": min(int(pitch_width), subject_pos[1] + 1000),
            },
            "is_subject": True,
        }
    ] + teammates_raw

    opponent_formation = [
        ("CB", (2200, 2500)),
        ("CB", (2200, 4500)),
        ("LB", (2200, 800)),
        ("RB", (2200, 6200)),
        ("CM", (5500, 2500)),
        ("CM", (5500, 3500)),
        ("CM", (5500, 4500)),
        ("LW", (9000, 1000)),
        ("ST", (9000, 3500)),
        ("RW", (9000, 6000)),
        ("GK", (500, 3500)),
    ]
    opponents = _place_players(opponent_formation, flip_x=True)

    total_dist = 0.0
    avg_x = 0.0
    if subject_rows:
        xs = [float(r["pitch_x_m"]) for r in subject_rows]
        avg_x = sum(xs) / len(xs)
    if len(subject_rows) >= 2:
        prev = None
        for r in subject_rows:
            x = float(r["pitch_x_m"])
            y = float(r["pitch_y_m"])
            if prev is not None:
                total_dist += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
    total_dist_m = total_dist / 100.0

    interpretation = (
        f"Based on observed movement patterns, the player shows {role.lower()} tendencies "
        f"with an average pitch position at ({avg_x:.0f} cm, {subject_avg_y_cm:.0f} cm). "
        f"Total distance covered: {total_dist_m:.1f} m. "
        f"Positioning suggests involvement in {'attacking' if avg_x > pitch_length * 0.6 else 'defensive'} phases. "
        f"Note: Teammate and opponent positions are SIMULATED based on standard tactical formations."
    )

    return {
        "simulation": {
            "subject_position": role,
            "teammates": teammates,
            "opponents": opponents,
            "tactical_interpretation": interpretation,
            "disclaimer": (
                "SIMULATED_DATA - Teammate and opponent positions are generated based on "
                "football tactical rules and do not reflect actual observed data. Only the "
                "subject player's position and movement are based on real CV analysis."
            ),
        }
    }


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

def _read_video_fps(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 25.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 25.0


class CVIntegrationService:
    """Service that wraps football-player-detection-main for individual player analysis."""

    def __init__(
        self,
        cv_repo_path: str | Path = CV_REPO_DIR,
        models_dir: str | Path | None = None,
        device: str | None = None,
        tracker_cfg: str = "bytetrack",
    ) -> None:
        self.cv_repo_path = Path(cv_repo_path)
        self.models_dir = Path(models_dir) if models_dir else CV_MODELS_DIR
        self.device = _resolve_device(device if device is not None else settings.device)
        self.tracker_cfg_name = tracker_cfg

        if not self.cv_repo_path.is_dir():
            raise FileNotFoundError(f"CV repository not found: {self.cv_repo_path}")
        if not CV_AVAILABLE:
            raise RuntimeError("CV modules could not be imported")

        self.pitch_cfg = SoccerPitchConfiguration()
        logger.info("CVIntegrationService initialized on device=%s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        enable_team_assignment: bool = False,
        enable_possession: bool = True,
        vid_stride: int = 1,
        heatmap_top_n: int = 1,
        generate_heatmap: bool = True,
        generate_trajectory: bool = True,
    ) -> dict[str, Any]:
        """Run full individual player analysis on a video.

        Returns a dict matching the required output format.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []

        if not video_path.is_file():
            return {
                "status": "failed",
                "subject_track_id": None,
                "selection_method": None,
                "frames_sampled": 0,
                "overlay_data": {},
                "metrics": {},
                "heatmap_points": [],
                "trajectory_points": [],
                "events": [],
                "simulation": {},
                "warnings": [f"Video not found: {video_path}"],
                "cv_repo_used": str(self.cv_repo_path),
            }

        try:
            tracker_weights = self._resolve_model("best_players_gk_1280_s_e300.pt")
            ball_weights = self._resolve_model("ball_tracking_1280_e300.pt")
            pitch_weights = self._resolve_model("pitch_kpts32_y8s_640_e500_AO.pt")
            tracker_cfg_path = self._resolve_tracker_cfg()
        except FileNotFoundError as exc:
            logger.error("Model file missing: %s", exc)
            return {
                "status": "failed",
                "subject_track_id": None,
                "selection_method": None,
                "frames_sampled": 0,
                "overlay_data": {},
                "metrics": {},
                "heatmap_points": [],
                "trajectory_points": [],
                "events": [],
                "simulation": {},
                "warnings": [f"Model file missing: {exc}"],
                "cv_repo_used": str(self.cv_repo_path),
            }

        tracker = CVTracker(
            weights=str(tracker_weights),
            device=self.device,
            image_size=1280,
            tracker_cfg=str(tracker_cfg_path),
        )
        pitch_detector = PitchKeypointDetector(
            weights=str(pitch_weights),
            device=self.device,
            image_size=640,
            num_keypoints=32,
        )
        ball_overlay_fn = (
            _build_ball_overlay_fn(ball_weights, self.device)
            if enable_possession
            else None
        )

        yolo_video_path = output_dir / "yolo_tracking.mp4"
        team_video_path = output_dir / "tactical_view.mp4"
        csv_output_path = output_dir / "tracks.csv"

        heatmap_dir = output_dir / "heatmaps" if generate_heatmap else None
        trajectory_dir = output_dir / "trajectories" if generate_trajectory else None
        trajectory_views = {"allplayers", "team", "topk"} if generate_trajectory else None
        report_dir = output_dir if (generate_heatmap or generate_trajectory) else None

        logger.info("Running CV pipeline on %s", video_path)
        try:
            processed_frames = run_video_pipeline(
                video_path=video_path,
                tracker=tracker,
                pitch_detector=pitch_detector,
                yolo_video_path=yolo_video_path,
                team_homography_video_path=team_video_path,
                vid_stride=vid_stride,
                enable_team_assignment=enable_team_assignment,
                ball_overlay_fn=ball_overlay_fn,
                enable_possession=enable_possession,
                heatmap_output_dir=heatmap_dir,
                heatmap_top_n=heatmap_top_n,
                trajectory_output_dir=trajectory_dir,
                trajectory_views=trajectory_views,
                trajectory_top_k=1,
                report_output_dir=report_dir,
                csv_output_path=csv_output_path,
            )
        except Exception as exc:
            logger.exception("Pipeline failed")
            return {
                "status": "failed",
                "subject_track_id": None,
                "selection_method": None,
                "frames_sampled": 0,
                "overlay_data": {},
                "metrics": {},
                "heatmap_points": [],
                "trajectory_points": [],
                "events": [],
                "simulation": {},
                "warnings": [f"Pipeline failed: {exc}"],
                "cv_repo_used": str(self.cv_repo_path),
            }

        if processed_frames == 0:
            return {
                "status": "failed",
                "subject_track_id": None,
                "selection_method": None,
                "frames_sampled": 0,
                "overlay_data": {},
                "metrics": {},
                "heatmap_points": [],
                "trajectory_points": [],
                "events": [],
                "simulation": {},
                "warnings": ["No frames processed by pipeline"],
                "cv_repo_used": str(self.cv_repo_path),
            }

        if not csv_output_path.is_file():
            return {
                "status": "failed",
                "subject_track_id": None,
                "selection_method": None,
                "frames_sampled": 0,
                "overlay_data": {},
                "metrics": {},
                "heatmap_points": [],
                "trajectory_points": [],
                "events": [],
                "simulation": {},
                "warnings": ["CSV output not generated by pipeline"],
                "cv_repo_used": str(self.cv_repo_path),
            }

        rows = _read_tracks_csv(csv_output_path)
        try:
            subject_tid, selection_method = _identify_subject_track_id(rows)
        except ValueError as exc:
            logger.error("Subject identification failed: %s", exc)
            return {
                "status": "failed",
                "subject_track_id": None,
                "selection_method": None,
                "frames_sampled": 0,
                "overlay_data": {},
                "metrics": {},
                "heatmap_points": [],
                "trajectory_points": [],
                "events": [],
                "simulation": {},
                "warnings": [str(exc)],
                "cv_repo_used": str(self.cv_repo_path),
            }

        player_tids = {
            int(r["track_id"]) for r in rows
            if r.get("cls_name") in {"Player", "Goalkeeper"} and int(r["track_id"]) >= 0
        }
        if len(player_tids) > 1:
            warnings.append(
                f"Multiple players detected ({len(player_tids)} tracks). "
                f"Auto-selected subject track_id={subject_tid} by longest track."
            )

        fps = _read_video_fps(video_path)

        metrics = _compute_subject_metrics(rows, subject_tid, fps=fps, pitch_cfg=self.pitch_cfg)
        subject_rows_all = [
            r for r in rows if int(r["track_id"]) == subject_tid
        ]
        subject_rows_all.sort(key=lambda r: int(r["frame"]))

        # Build overlay data sampled every 5th frame
        subject_rows_by_frame = {
            int(r["frame"]): r for r in subject_rows_all
        }
        ball_rows_by_frame: dict[int, dict[str, str]] = {}
        for r in rows:
            if int(r["track_id"]) == -1 and r.get("cls_name") == "Ball":
                ball_rows_by_frame[int(r["frame"])] = r

        # Build per-frame player lookup for possession
        frame_players: dict[int, list[dict[str, str]]] = defaultdict(list)
        for r in rows:
            if r.get("cls_name") in {"Player", "Goalkeeper"} and int(r["track_id"]) >= 0:
                frame_players[int(r["frame"])].append(r)

        overlay_data: dict[str, Any] = {}
        sampled_count = 0
        sampled_frames: list[int] = []
        all_frame_nums = sorted(set(int(r["frame"]) for r in rows))
        for i, frame_num in enumerate(all_frame_nums):
            if i % 5 == 0:
                sampled_frames.append(frame_num)

        for frame_num in sampled_frames:
            s_row = subject_rows_by_frame.get(frame_num)
            b_row = ball_rows_by_frame.get(frame_num)
            timestamp_s = round(frame_num / fps, 3)

            player_dict: Optional[dict[str, Any]] = None
            pitch_dict: Optional[dict[str, float]] = None
            speed_kmh: Optional[float] = None
            ball_dict: Optional[dict[str, float]] = None
            possession = False
            event: Optional[str] = None

            if s_row:
                x1 = float(s_row["x1"])
                y1 = float(s_row["y1"])
                x2 = float(s_row["x2"])
                y2 = float(s_row["y2"])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                player_dict = {
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": round(float(s_row["conf"]), 4),
                    "track_id": subject_tid,
                }
                px = s_row.get("pitch_x_m")
                py = s_row.get("pitch_y_m")
                if px and float(px) > 0:
                    pitch_dict = {
                        "x": round(float(px), 2),
                        "y": round(float(py), 2),
                    }

            if b_row:
                bx1 = float(b_row["x1"])
                by1 = float(b_row["y1"])
                bx2 = float(b_row["x2"])
                by2 = float(b_row["y2"])
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                ball_dict = {"x": round(bcx, 2), "y": round(bcy, 2)}

            # Possession: closest player to ball within threshold
            if ball_dict and player_dict:
                min_dist = float("inf")
                closest_tid = None
                for fp in frame_players.get(frame_num, []):
                    fcx = (float(fp["x1"]) + float(fp["x2"])) / 2
                    fcy = (float(fp["y1"]) + float(fp["y2"])) / 2
                    d = math.hypot(fcx - ball_dict["x"], fcy - ball_dict["y"])
                    if d < min_dist:
                        min_dist = d
                        closest_tid = int(fp["track_id"])
                possession = closest_tid == subject_tid and min_dist < 150

            # Speed from consecutive sampled frames
            idx = sampled_frames.index(frame_num) if frame_num in sampled_frames else -1
            if idx > 0:
                prev_frame_num = sampled_frames[idx - 1]
                prev_row = subject_rows_by_frame.get(prev_frame_num)
                if prev_row and s_row:
                    prev_x = float(prev_row["pitch_x_m"])
                    prev_y = float(prev_row["pitch_y_m"])
                    cur_x = float(s_row["pitch_x_m"])
                    cur_y = float(s_row["pitch_y_m"])
                    dist_cm = math.hypot(cur_x - prev_x, cur_y - prev_y)
                    dt_s = (frame_num - prev_frame_num) / fps if fps > 0 else 0.04
                    if dt_s > 0:
                        speed_mps = (dist_cm / 100.0) / dt_s
                        speed_kmh = round(speed_mps * 3.6, 2)

            overlay_data[str(frame_num)] = {
                "t": timestamp_s,
                "player": player_dict,
                "pitch_position": pitch_dict,
                "speed_kmh": speed_kmh,
                "ball": ball_dict,
                "possession": possession,
                "event": event,
            }
            sampled_count += 1

        events = _detect_events(subject_rows_all, fps=fps)

        heatmap_points = self._extract_heatmap_points(rows, subject_tid)
        trajectory_points = self._extract_trajectory_points(rows, subject_tid)

        avg_x_cm = avg_y_cm = 0.0
        if subject_rows_all:
            xs = [float(r["pitch_x_m"]) for r in subject_rows_all if r.get("pitch_x_m")]
            ys = [float(r["pitch_y_m"]) for r in subject_rows_all if r.get("pitch_y_m")]
            if xs:
                avg_x_cm = sum(xs) / len(xs)
            if ys:
                avg_y_cm = sum(ys) / len(ys)

        simulation = _generate_simulation(
            subject_tid=subject_tid,
            subject_avg_x_cm=avg_x_cm,
            subject_avg_y_cm=avg_y_cm,
            subject_rows=subject_rows_all,
            pitch_cfg=self.pitch_cfg,
            fps=fps,
        )

        pitch_valid_rows = [r for r in rows if r.get("homography_valid") == "1"]
        if not pitch_valid_rows:
            warnings.append(
                "Pitch keypoints not reliably detected; tactical data is limited to pixel space."
            )

        ball_detected = any(int(r["track_id"]) == -1 for r in rows)
        if not ball_detected and enable_possession:
            warnings.append(
                "Ball not detected in sufficient frames; ball metrics are limited."
            )

        if metrics.get("confidence") == "low":
            warnings.append(
                "Insufficient tracking data for high-confidence metrics."
            )

        status = "completed" if metrics.get("total_distance_m") is not None else "partial"

        return {
            "status": status,
            "subject_track_id": subject_tid,
            "selection_method": selection_method,
            "frames_sampled": sampled_count,
            "overlay_data": overlay_data,
            "metrics": metrics,
            "heatmap_points": heatmap_points,
            "trajectory_points": trajectory_points,
            "events": events,
            "simulation": simulation,
            "warnings": warnings,
            "cv_repo_used": str(self.cv_repo_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, name: str) -> Path:
        p = self.models_dir / name
        if not p.is_file():
            p = self.cv_repo_path / "models" / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Model not found: {name} (searched {self.models_dir} and {self.cv_repo_path / 'models'})"
            )
        return p

    def _resolve_tracker_cfg(self) -> Path:
        name = f"{self.tracker_cfg_name}.yaml"
        p = CV_CONFIGS_DIR / name
        if not p.is_file():
            raise FileNotFoundError(f"Tracker config not found: {p}")
        return p

    def _extract_heatmap_points(
        self, rows: list[dict[str, str]], subject_tid: int
    ) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        for r in rows:
            if int(r["track_id"]) == subject_tid:
                px = r.get("pitch_x_m")
                py = r.get("pitch_y_m")
                if px and py and float(px) > 0:
                    points.append({"x": round(float(px), 2), "y": round(float(py), 2)})
        return points

    def _extract_trajectory_points(
        self, rows: list[dict[str, str]], subject_tid: int
    ) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        subject_rows = [
            r for r in rows
            if int(r["track_id"]) == subject_tid
            and r.get("pitch_point_valid") == "1"
        ]
        subject_rows.sort(key=lambda r: int(r["frame"]))
        for r in subject_rows:
            x1 = float(r["x1"])
            y1 = float(r["y1"])
            x2 = float(r["x2"])
            y2 = float(r["y2"])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            points.append({
                "frame": int(r["frame"]),
                "pixel": {"x": round(cx, 2), "y": round(cy, 2)},
                "pitch": {
                    "x": round(float(r["pitch_x_m"]), 2),
                    "y": round(float(r["pitch_y_m"]), 2),
                },
            })
        return points
