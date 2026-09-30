from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import supervision as sv

from src.analytics.homography_simple import HomographyTransformer
from src.analytics.pass_detection import PassDetector
from src.analytics.possession import PossessionTracker, TeamPossessionTracker
from src.analytics.team_assignment import TeamAssigner
from src.core.pitch import SoccerPitchConfiguration
from src.core.types import Track
from src.io.csv_writer import CsvWriter
from src.io.minimap_renderer import MinimapRenderer
from src.vision.pitch_keypoint_detector import PitchKeypointDetector
from src.vision.tracker import Tracker


TRACKED_CLASSES = {"Player", "Goalkeeper", "Referee"}

# Palette: Team 0, Team 1, Referee, Unassigned/Goalkeeper-no-team
_TEAM_COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
# Non-team mode: Player, Goalkeeper, Referee, fallback
_CLASS_COLORS = ["#00BFFF", "#32CD32", "#FF6347", "#C0C0C0"]

_ELLIPSE_ANNOTATOR_TEAM = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(_TEAM_COLORS),
    thickness=2,
)
_LABEL_ANNOTATOR_TEAM = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(_TEAM_COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)
_ELLIPSE_ANNOTATOR_CLASS = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(_CLASS_COLORS),
    thickness=2,
)
_LABEL_ANNOTATOR_CLASS = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(_CLASS_COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


@dataclass(frozen=True)
class HomographyFrameStatus:
    status: str
    reason: str
    stale_frames: int
    num_points: int


def _read_video_metadata(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video dimensions from: {video_path}")
    if fps <= 0:
        fps = 25.0
    if total_frames <= 0:
        total_frames = -1
    return fps, width, height, total_frames


def _build_pitch_world_points(pitch_cfg: SoccerPitchConfiguration) -> np.ndarray:
    # Match Roboflow's radar example: use config vertices in declared order.
    return np.asarray(pitch_cfg.vertices, dtype=np.float32)


def _draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    include_team: bool,
    team_colored: bool,
) -> None:
    """Annotate tracks using supervision EllipseAnnotator + LabelAnnotator."""
    track_list = list(tracks)
    if not track_list:
        return

    xyxy = np.array([t.bbox for t in track_list], dtype=np.float32)
    tracker_ids = np.array(
        [t.track_id if t.track_id >= 0 else -1 for t in track_list], dtype=int
    )

    detections = sv.Detections(xyxy=xyxy)
    detections.tracker_id = tracker_ids

    # Build labels and color lookup.
    labels: list[str] = []
    color_lookup: list[int] = []

    for track in track_list:
        tid = track.track_id if track.track_id >= 0 else -1
        if team_colored:
            # Team palette: 0=Team0, 1=Team1, 2=Referee, 3=Unassigned
            if track.cls_name == "Referee":
                color_lookup.append(2)
                labels.append(str(tid) if tid >= 0 else "REF")
            elif track.cls_name in {"Player", "Goalkeeper"}:
                if track.team_id is not None:
                    color_lookup.append(int(track.team_id) % 2)
                else:
                    color_lookup.append(3)
                labels.append(str(tid) if tid >= 0 else "?")
            else:
                color_lookup.append(3)
                labels.append(str(tid) if tid >= 0 else "?")
        else:
            # Class palette: 0=Player, 1=Goalkeeper, 2=Referee, 3=Other
            if track.cls_name == "Player":
                color_lookup.append(0)
            elif track.cls_name == "Goalkeeper":
                color_lookup.append(1)
            elif track.cls_name == "Referee":
                color_lookup.append(2)
            else:
                color_lookup.append(3)
            labels.append(str(tid) if tid >= 0 else track.cls_name[:3])

    color_lookup_arr = np.array(color_lookup, dtype=int)

    if team_colored:
        ellipse_ann = _ELLIPSE_ANNOTATOR_TEAM
        label_ann = _LABEL_ANNOTATOR_TEAM
    else:
        ellipse_ann = _ELLIPSE_ANNOTATOR_CLASS
        label_ann = _LABEL_ANNOTATOR_CLASS

    ellipse_ann.annotate(frame, detections, custom_color_lookup=color_lookup_arr)
    label_ann.annotate(
        frame, detections, labels=labels, custom_color_lookup=color_lookup_arr
    )


def _draw_possession_indicator(
    frame: np.ndarray,
    tracks: Iterable[Track],
    possessor_id: int | None,
) -> None:
    """Draw a FIFA-style downward triangle above the possessor's bounding box."""
    if possessor_id is None:
        return
    for track in tracks:
        if track.track_id == possessor_id:
            x1, y1, x2, y2 = track.bbox
            cx = int((x1 + x2) / 2)
            top = int(y1) - 28
            pts = np.array([[cx, top + 16], [cx - 8, top], [cx + 8, top]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 255))
            cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
            break


def _draw_possession_bar(
    panel: np.ndarray,
    panel_height: int,
    frame_width: int,
    percentages: dict[int, float],
) -> None:
    """Draw a horizontal team possession bar on the right side of the panel."""
    if not percentages:
        return

    pct0 = percentages.get(0, 0.0)
    pct1 = percentages.get(1, 0.0)
    total = pct0 + pct1
    if total < 0.01:
        return

    # Normalize to sum to 100
    pct0 = pct0 / total * 100.0
    pct1 = pct1 / total * 100.0

    bar_w = 300
    bar_h = 30
    # Position: right side of the panel, vertically centered
    bar_x = frame_width - bar_w - 100
    bar_y = (panel_height - bar_h) // 2

    # Team 0 (pink #FF1493 → BGR: 147, 20, 255), Team 1 (cyan #00BFFF → BGR: 255, 191, 0)
    color0 = (147, 20, 255)
    color1 = (255, 191, 0)

    split_x = int(bar_w * pct0 / 100.0)

    # Draw bar segments
    if split_x > 0:
        cv2.rectangle(
            panel,
            (bar_x, bar_y),
            (bar_x + split_x, bar_y + bar_h),
            color0,
            -1,
        )
    if split_x < bar_w:
        cv2.rectangle(
            panel,
            (bar_x + split_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            color1,
            -1,
        )

    # Border
    cv2.rectangle(
        panel,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (60, 60, 60),
        1,
        lineType=cv2.LINE_AA,
    )

    # Percentage labels
    label0 = f"{pct0:.0f}%"
    label1 = f"{pct1:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Team 0 label (left of bar)
    sz0 = cv2.getTextSize(label0, font, font_scale, thickness)[0]
    cv2.putText(
        panel,
        label0,
        (bar_x - sz0[0] - 6, bar_y + bar_h // 2 + sz0[1] // 2),
        font,
        font_scale,
        color0,
        thickness,
        lineType=cv2.LINE_AA,
    )

    # Team 1 label (right of bar)
    cv2.putText(
        panel,
        label1,
        (bar_x + bar_w + 6, bar_y + bar_h // 2 + sz0[1] // 2),
        font,
        font_scale,
        color1,
        thickness,
        lineType=cv2.LINE_AA,
    )

    # "Possession" label above the bar
    poss_sz = cv2.getTextSize("Possession", font, 0.5, 1)[0]
    cv2.putText(
        panel,
        "Possession",
        (bar_x + bar_w // 2 - poss_sz[0] // 2, bar_y - 8),
        font,
        0.5,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )


def _draw_pass_counts(
    panel: np.ndarray,
    panel_height: int,
    frame_width: int,
    pass_summary: dict[int, dict[str, int]],
) -> None:
    """Draw team pass totals below the possession bar in a matching style."""
    color0 = (147, 20, 255)  # pink BGR
    color1 = (255, 191, 0)  # cyan BGR
    font = cv2.FONT_HERSHEY_SIMPLEX

    count0 = pass_summary.get(0, {}).get("completed", 0)
    count1 = pass_summary.get(1, {}).get("completed", 0)

    # Match the possession bar's horizontal position
    bar_w = 300
    bar_h = 30
    bar_x = frame_width - bar_w - 100
    # Possession bar is at y = (panel_height - bar_h) // 2
    # Place passes bar below it with some spacing
    poss_bar_y = (panel_height - bar_h) // 2
    pass_y = poss_bar_y + bar_h + 30

    # "Passes" label above the bar
    hdr_sz = cv2.getTextSize("Passes", font, 0.5, 1)[0]
    cv2.putText(
        panel,
        "Passes",
        (bar_x + bar_w // 2 - hdr_sz[0] // 2, pass_y - 8),
        font,
        0.5,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )

    # Draw split bar — each half shows the team count
    half_w = bar_w // 2
    gap = 2  # small gap between the two halves

    # Team 0 box (left half)
    cv2.rectangle(
        panel,
        (bar_x, pass_y),
        (bar_x + half_w - gap, pass_y + bar_h),
        color0,
        -1,
    )
    # Team 1 box (right half)
    cv2.rectangle(
        panel,
        (bar_x + half_w + gap, pass_y),
        (bar_x + bar_w, pass_y + bar_h),
        color1,
        -1,
    )

    # Count labels centered in each half
    label0 = str(count0)
    label1 = str(count1)
    font_scale = 0.65
    thickness = 2
    text_color = (255, 255, 255)

    sz0 = cv2.getTextSize(label0, font, font_scale, thickness)[0]
    sz1 = cv2.getTextSize(label1, font, font_scale, thickness)[0]

    cv2.putText(
        panel,
        label0,
        (
            bar_x + (half_w - gap) // 2 - sz0[0] // 2,
            pass_y + bar_h // 2 + sz0[1] // 2,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        label1,
        (
            bar_x + half_w + gap + (half_w - gap) // 2 - sz1[0] // 2,
            pass_y + bar_h // 2 + sz1[1] // 2,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_turnover_counts(
    panel: np.ndarray,
    panel_height: int,
    frame_width: int,
    turnover_summary: dict[int, dict[str, int]],
) -> None:
    """Draw team turnover totals below the passes bar in a matching style."""
    color0 = (147, 20, 255)  # pink BGR
    color1 = (255, 191, 0)  # cyan BGR
    font = cv2.FONT_HERSHEY_SIMPLEX

    count0 = turnover_summary.get(0, {}).get("lost", 0)
    count1 = turnover_summary.get(1, {}).get("lost", 0)

    bar_w = 300
    bar_h = 30
    bar_x = frame_width - bar_w - 100
    poss_bar_y = (panel_height - bar_h) // 2
    pass_y = poss_bar_y + bar_h + 30
    turnover_y = pass_y + bar_h + 30

    # "Turnovers" label above the bar
    hdr_sz = cv2.getTextSize("Turnovers", font, 0.5, 1)[0]
    cv2.putText(
        panel,
        "Turnovers",
        (bar_x + bar_w // 2 - hdr_sz[0] // 2, turnover_y - 8),
        font,
        0.5,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )

    half_w = bar_w // 2
    gap = 2

    # Team 0 box (left half)
    cv2.rectangle(
        panel,
        (bar_x, turnover_y),
        (bar_x + half_w - gap, turnover_y + bar_h),
        color0,
        -1,
    )
    # Team 1 box (right half)
    cv2.rectangle(
        panel,
        (bar_x + half_w + gap, turnover_y),
        (bar_x + bar_w, turnover_y + bar_h),
        color1,
        -1,
    )

    label0 = str(count0)
    label1 = str(count1)
    font_scale = 0.65
    thickness = 2
    text_color = (255, 255, 255)

    sz0 = cv2.getTextSize(label0, font, font_scale, thickness)[0]
    sz1 = cv2.getTextSize(label1, font, font_scale, thickness)[0]

    cv2.putText(
        panel,
        label0,
        (
            bar_x + (half_w - gap) // 2 - sz0[0] // 2,
            turnover_y + bar_h // 2 + sz0[1] // 2,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        label1,
        (
            bar_x + half_w + gap + (half_w - gap) // 2 - sz1[0] // 2,
            turnover_y + bar_h // 2 + sz1[1] // 2,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _status_text(h_status: HomographyFrameStatus) -> str:
    if h_status.status == "valid":
        return f"H: valid ({h_status.num_points} pts)"
    if h_status.status == "fallback":
        return f"H: fallback {h_status.stale_frames} ({h_status.reason})"
    return f"H: invalid ({h_status.reason})"


def _format_diag_float(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "na"


def _should_render_keypoint_debug(
    *,
    frame_idx: int,
    enabled: bool,
    start_frame: Optional[int],
    end_frame: Optional[int],
    last_n_frames: int,
    total_output_frames: Optional[int],
) -> bool:
    if not enabled:
        return False

    if start_frame is not None or end_frame is not None:
        start = 1 if start_frame is None else start_frame
        end = 10**9 if end_frame is None else end_frame
        return start <= frame_idx <= end

    if last_n_frames <= 0:
        return False

    if total_output_frames is None or total_output_frames <= 0:
        # If total frame count is unknown, show on all processed frames.
        return True

    start_last = max(1, total_output_frames - last_n_frames + 1)
    return frame_idx >= start_last


def _draw_pitch_keypoint_debug(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    conf_threshold: float,
) -> None:
    valid_count = 0

    for idx, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf), start=1):
        if not np.isfinite(xy).all():
            continue

        x_i = int(round(float(xy[0])))
        y_i = int(round(float(xy[1])))
        if x_i <= 1 or y_i <= 1:
            continue

        is_valid = float(conf) >= conf_threshold
        if is_valid:
            valid_count += 1
            color = (80, 220, 80)  # green
            radius = 4
            thickness = 2
        else:
            color = (60, 120, 255)  # orange
            radius = 3
            thickness = 1

        cv2.circle(frame, (x_i, y_i), radius, color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            str(idx),
            (x_i + 4, y_i - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"KPTS {valid_count}/{len(keypoints_xy)}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        lineType=cv2.LINE_AA,
    )


def run_video_pipeline(
    *,
    video_path: Path,
    tracker: Tracker,
    pitch_detector: PitchKeypointDetector,
    yolo_video_path: Path,
    team_homography_video_path: Path,
    vid_stride: int = 1,
    homography_keypoint_conf_threshold: float = 0.3,
    homography_min_points: int = 4,
    homography_max_stale_frames: int = 60,
    homography_min_inliers: int = 4,
    homography_min_inlier_ratio: float = 0.5,
    homography_min_inlier_x_span_cm: float = 1200.0,
    homography_min_inlier_y_span_cm: float = 1000.0,
    homography_max_reproj_error_cm: float = 120.0,
    enable_team_assignment: bool = True,
    ball_overlay_fn: Optional[
        Callable[[np.ndarray], tuple[np.ndarray, tuple[float, float] | None]]
    ] = None,
    enable_possession: bool = True,
    debug_keypoints: bool = False,
    debug_homography_diag: bool = False,
    debug_keypoints_last_n: int = 100,
    debug_keypoints_start_frame: Optional[int] = None,
    debug_keypoints_end_frame: Optional[int] = None,
    position_smoothing_alpha: float = 0.85,
    max_position_jump_cm: float = 250.0,
    position_state_stale_frames: int = 90,
    heatmap_output_dir: Optional[Path] = None,
    heatmap_top_n: int = 0,
    heatmap_track_ids: Optional[list[int]] = None,
    trajectory_output_dir: Optional[Path] = None,
    trajectory_views: Optional[set[str]] = None,
    trajectory_top_k: int = 3,
    trajectory_track_ids: Optional[list[int]] = None,
    report_output_dir: Optional[Path] = None,
    pass_network_output_dir: Optional[Path] = None,
    csv_output_path: Optional[Path] = None,
) -> int:
    if vid_stride < 1:
        raise ValueError("vid_stride must be >= 1")
    if not (0.0 <= position_smoothing_alpha < 1.0):
        raise ValueError("position_smoothing_alpha must be in [0, 1)")
    if max_position_jump_cm <= 0:
        raise ValueError("max_position_jump_cm must be > 0")
    if position_state_stale_frames < 1:
        raise ValueError("position_state_stale_frames must be >= 1")
    if homography_min_inliers < 4:
        raise ValueError("homography_min_inliers must be >= 4")
    if not (0.0 <= homography_min_inlier_ratio <= 1.0):
        raise ValueError("homography_min_inlier_ratio must be in [0, 1]")
    if homography_min_inlier_x_span_cm <= 0:
        raise ValueError("homography_min_inlier_x_span_cm must be > 0")
    if homography_min_inlier_y_span_cm <= 0:
        raise ValueError("homography_min_inlier_y_span_cm must be > 0")
    if homography_max_reproj_error_cm <= 0:
        raise ValueError("homography_max_reproj_error_cm must be > 0")

    yolo_video_path.parent.mkdir(parents=True, exist_ok=True)
    team_homography_video_path.parent.mkdir(parents=True, exist_ok=True)

    fps, width, height, total_input_frames = _read_video_metadata(video_path)
    total_output_frames: Optional[int] = None
    if total_input_frames > 0:
        total_output_frames = int(math.ceil(total_input_frames / vid_stride))
    radar_panel_height = max(280, height // 3)
    yolo_writer = cv2.VideoWriter(
        str(yolo_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    tactical_writer = cv2.VideoWriter(
        str(team_homography_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height + radar_panel_height),
    )
    if not yolo_writer.isOpened():
        raise RuntimeError(f"Could not open YOLO output video: {yolo_video_path}")
    if not tactical_writer.isOpened():
        raise RuntimeError(
            f"Could not open tactical output video: {team_homography_video_path}"
        )

    pitch_cfg = SoccerPitchConfiguration()
    pitch_world_points = _build_pitch_world_points(pitch_cfg)
    minimap_renderer = MinimapRenderer(
        pitch_length_m=float(pitch_cfg.length),
        pitch_width_m=float(pitch_cfg.width),
        units_per_meter=100.0,
    )

    team_assigner = TeamAssigner()

    homography_transformer: Optional[HomographyTransformer] = None
    homography_stale_frames = 0
    processed_frames = 0
    smoothed_pitch_positions: dict[int, np.ndarray] = {}
    last_seen_pitch_frame: dict[int, int] = {}

    _want_heatmaps = heatmap_output_dir is not None and (
        heatmap_top_n > 0 or heatmap_track_ids
    )
    _want_trajectories = trajectory_output_dir is not None
    _want_report = report_output_dir is not None
    _want_pass_network = pass_network_output_dir is not None
    _want_accumulate = (
        _want_heatmaps or _want_trajectories or _want_report or _want_pass_network
    )
    accum_positions: dict[int, list[tuple[int, float, float]]] = {}
    accum_metadata: dict[int, tuple[str, int | None]] = {}
    possession_tracker = PossessionTracker() if enable_possession else None
    team_possession_tracker = (
        TeamPossessionTracker()
        if enable_possession and enable_team_assignment
        else None
    )
    pass_detector = (
        PassDetector() if enable_possession and enable_team_assignment else None
    )
    csv_writer = CsvWriter(str(csv_output_path)) if csv_output_path is not None else None

    ball_pos_buffer: deque[tuple[float, float] | None] = deque(maxlen=5)
    effective_dt = vid_stride / fps
    ball_speed_kmh: float | None = None

    # Pass arrows for minimap: (passer_pos_cm, receiver_pos_cm, team_id, created_frame)
    _pass_arrow_duration = 30  # frames (~1 sec at 30fps)
    _active_pass_arrows: list[
        tuple[tuple[float, float], tuple[float, float], int, int]
    ] = []

    try:
        for frame_data in tracker.frames(
            video_path=str(video_path),
            vid_stride=vid_stride,
            save=False,
        ):
            processed_frames += 1
            tracks = [t for t in frame_data.tracks if t.cls_name in TRACKED_CLASSES]

            if enable_team_assignment:
                if frame_data.frame_idx <= 60 and frame_data.frame_idx % 5 == 0:
                    team_assigner.fit(tracks, frame_data.img)
                if frame_data.frame_idx == 61:
                    team_assigner.train_model()
                if frame_data.frame_idx > 60:
                    team_assigner.predict(frame_data.img, tracks)

            keypoint_result = pitch_detector.detect(frame_data.img)
            n = min(
                len(keypoint_result.xy),
                len(keypoint_result.conf),
                len(pitch_world_points),
            )
            keypoints_xy = keypoint_result.xy[:n]
            keypoints_conf = keypoint_result.conf[:n]

            conf_mask = keypoints_conf >= homography_keypoint_conf_threshold
            finite_mask = np.isfinite(keypoints_xy).all(axis=1)
            in_bounds_mask = (keypoints_xy[:, 0] > 1.0) & (keypoints_xy[:, 1] > 1.0)
            in_bounds_mask &= keypoints_xy[:, 0] < float(frame_data.width - 2)
            in_bounds_mask &= keypoints_xy[:, 1] < float(frame_data.height - 2)

            valid_mask = conf_mask & finite_mask & in_bounds_mask

            src = keypoints_xy[valid_mask].astype(np.float32)
            dst = pitch_world_points[:n][valid_mask].astype(np.float32)
            num_points = int(len(src))
            conf_count = int(conf_mask.sum())
            finite_count = int(finite_mask.sum())
            in_bounds_count = int(in_bounds_mask.sum())

            if num_points < homography_min_points:
                if (
                    homography_transformer is not None
                    and homography_stale_frames < homography_max_stale_frames
                ):
                    homography_stale_frames += 1
                    h_status = HomographyFrameStatus(
                        status="fallback",
                        reason="low_points",
                        stale_frames=homography_stale_frames,
                        num_points=num_points,
                    )
                else:
                    homography_stale_frames += 1
                    homography_transformer = None
                    h_status = HomographyFrameStatus(
                        status="invalid",
                        reason="low_points",
                        stale_frames=homography_stale_frames,
                        num_points=num_points,
                    )
                    print(f"Low points: {num_points}")
            else:
                try:
                    candidate_transformer = HomographyTransformer(
                        source=src, target=dst
                    )
                except ValueError:
                    if (
                        homography_transformer is not None
                        and homography_stale_frames < homography_max_stale_frames
                    ):
                        homography_stale_frames += 1
                        h_status = HomographyFrameStatus(
                            status="fallback",
                            reason="no_homography_solution",
                            stale_frames=homography_stale_frames,
                            num_points=num_points,
                        )
                    else:
                        homography_stale_frames += 1
                        homography_transformer = None
                        h_status = HomographyFrameStatus(
                            status="invalid",
                            reason="no_homography_solution",
                            stale_frames=homography_stale_frames,
                            num_points=num_points,
                        )
                else:
                    reproj = candidate_transformer.transform_points(src)
                    reproj_err = np.linalg.norm(reproj - dst, axis=1)
                    inlier_mask = reproj_err <= homography_max_reproj_error_cm
                    num_inliers = int(inlier_mask.sum())
                    inlier_ratio = float(num_inliers / max(num_points, 1))

                    if num_inliers > 0:
                        inlier_dst = dst[inlier_mask]
                        x_span_cm = float(
                            inlier_dst[:, 0].max() - inlier_dst[:, 0].min()
                        )
                        y_span_cm = float(
                            inlier_dst[:, 1].max() - inlier_dst[:, 1].min()
                        )
                        median_reproj_error_cm = float(
                            np.median(reproj_err[inlier_mask])
                        )
                    else:
                        x_span_cm = 0.0
                        y_span_cm = 0.0
                        median_reproj_error_cm = float("inf")

                    gate_fail = None
                    if num_inliers < homography_min_inliers:
                        gate_fail = "low_inliers"
                    elif inlier_ratio < homography_min_inlier_ratio:
                        gate_fail = "low_inlier_ratio"
                    elif x_span_cm < homography_min_inlier_x_span_cm:
                        gate_fail = "low_inlier_x_span"
                    elif y_span_cm < homography_min_inlier_y_span_cm:
                        gate_fail = "low_inlier_y_span"
                    elif median_reproj_error_cm > homography_max_reproj_error_cm:
                        gate_fail = "high_reproj_error"

                    if gate_fail is None:
                        homography_transformer = candidate_transformer
                        homography_stale_frames = 0
                        h_status = HomographyFrameStatus(
                            status="valid",
                            reason="accepted",
                            stale_frames=0,
                            num_points=num_points,
                        )
                    elif (
                        homography_transformer is not None
                        and homography_stale_frames < homography_max_stale_frames
                    ):
                        homography_stale_frames += 1
                        h_status = HomographyFrameStatus(
                            status="fallback",
                            reason=gate_fail,
                            stale_frames=homography_stale_frames,
                            num_points=num_points,
                        )
                    else:
                        homography_stale_frames += 1
                        homography_transformer = None
                        h_status = HomographyFrameStatus(
                            status="invalid",
                            reason=gate_fail,
                            stale_frames=homography_stale_frames,
                            num_points=num_points,
                        )

            if debug_homography_diag:
                print(
                    "H-DIAG "
                    f"frame={frame_data.frame_idx} "
                    f"raw_det={int(getattr(keypoint_result, 'num_raw_detections', 0) or 0)} "
                    f"bbox_conf={_format_diag_float(getattr(keypoint_result, 'bbox_conf', 0.0))} "
                    f"conf_ok={conf_count} "
                    f"finite={finite_count} "
                    f"in_bounds={in_bounds_count} "
                    f"valid={num_points} "
                    f"status={getattr(h_status, 'status', 'unknown')} "
                    f"reason={getattr(h_status, 'reason', 'unknown')}"
                )

            for track in tracks:
                track.pitch_x_cm = None
                track.pitch_y_cm = None

            if homography_transformer is not None and tracks:
                foot_points = np.asarray(
                    [t.foot_position for t in tracks], dtype=np.float32
                )
                projected_points = homography_transformer.transform_points(foot_points)
                for track, projected in zip(tracks, projected_points):
                    if np.isfinite(projected).all():
                        projected_xy = projected.astype(np.float32)
                        if track.track_id >= 0:
                            prev_xy = smoothed_pitch_positions.get(track.track_id)
                            if prev_xy is not None:
                                delta = projected_xy - prev_xy
                                dist = float(np.linalg.norm(delta))
                                if dist > max_position_jump_cm and dist > 1e-6:
                                    projected_xy = (
                                        prev_xy + (delta / dist) * max_position_jump_cm
                                    )
                                projected_xy = (
                                    position_smoothing_alpha * prev_xy
                                    + (1.0 - position_smoothing_alpha) * projected_xy
                                )
                            smoothed_pitch_positions[track.track_id] = projected_xy
                            last_seen_pitch_frame[track.track_id] = frame_data.frame_idx
                            projected_xy = smoothed_pitch_positions[track.track_id]

                        track.pitch_x_cm = float(projected_xy[0])
                        track.pitch_y_cm = float(projected_xy[1])

            # Keep only recent tracks in the smoothing cache.
            stale_ids = [
                track_id
                for track_id, seen_frame in last_seen_pitch_frame.items()
                if frame_data.frame_idx - seen_frame > position_state_stale_frames
            ]
            for track_id in stale_ids:
                last_seen_pitch_frame.pop(track_id, None)
                smoothed_pitch_positions.pop(track_id, None)

            if _want_accumulate:
                for track in tracks:
                    if track.pitch_x_cm is not None and track.pitch_y_cm is not None:
                        accum_positions.setdefault(track.track_id, []).append(
                            (frame_data.frame_idx, track.pitch_x_cm, track.pitch_y_cm)
                        )
                        accum_metadata[track.track_id] = (
                            track.cls_name,
                            track.team_id,
                        )

            ball_xy: tuple[float, float] | None = None
            yolo_frame = frame_data.img.copy()
            if ball_overlay_fn is not None:
                yolo_frame, ball_xy = ball_overlay_fn(yolo_frame)
            _draw_tracks(
                yolo_frame,
                tracks,
                include_team=False,
                team_colored=False,
            )

            # Ball velocity calculation
            ball_pos_buffer.append(ball_xy)
            if (
                len(ball_pos_buffer) >= 2
                and ball_pos_buffer[-1] is not None
                and ball_pos_buffer[-2] is not None
                and homography_transformer is not None
            ):
                pts = np.array(
                    [ball_pos_buffer[-2], ball_pos_buffer[-1]], dtype=np.float32
                )
                pitch_pts = homography_transformer.transform_points(pts)
                if np.isfinite(pitch_pts).all():
                    dist_cm = float(np.linalg.norm(pitch_pts[1] - pitch_pts[0]))
                    speed_cms = dist_cm / effective_dt
                    ball_speed_kmh = speed_cms * 0.036  # cm/s → km/h
                else:
                    ball_speed_kmh = None
            else:
                ball_speed_kmh = None

            if csv_writer is not None:
                # Build list with player/GK/ref tracks + synthetic ball track
                csv_tracks = list(tracks)
                if ball_xy is not None:
                    from src.core.types import Detection, Track as TrackCls

                    bx, by = ball_xy
                    ball_det = Detection(
                        bbox=(bx - 5, by - 5, bx + 5, by + 5),
                        conf=1.0,
                        cls_name="Ball",
                        cls_id=3,
                    )
                    ball_track = TrackCls(detection=ball_det, track_id=-1)
                    # Project ball to pitch if homography is available
                    if homography_transformer is not None:
                        ball_pitch = homography_transformer.transform_points(
                            np.array([[bx, by]], dtype=np.float32)
                        )
                        if np.isfinite(ball_pitch).all():
                            ball_track.pitch_x_cm = float(ball_pitch[0][0])
                            ball_track.pitch_y_cm = float(ball_pitch[0][1])
                    csv_tracks.append(ball_track)
                csv_writer.write_frame(
                    frame_idx=frame_data.frame_idx,
                    tracks=csv_tracks,
                    width=frame_data.width,
                    height=frame_data.height,
                    homography_valid=(h_status.status == "valid"),
                    homography_used_fallback=(h_status.status == "fallback"),
                    homography_stale_frames=h_status.stale_frames,
                    homography_status=h_status.status,
                    homography_reject_reason=h_status.reason,
                )

            possessor_id: int | None = None
            if possession_tracker is not None:
                possessor_id = possession_tracker.update(tracks, ball_xy)

            if team_possession_tracker is not None:
                team_possession_tracker.update(possessor_id, tracks)

            if pass_detector is not None:
                pass_event = pass_detector.update(
                    possessor_id, tracks, frame_data.frame_idx
                )
                if pass_event is not None and h_status.status == "valid":
                    _active_pass_arrows.append(
                        (
                            pass_event.passer_pos_cm,
                            pass_event.receiver_pos_cm,
                            pass_event.team_id,
                            frame_data.frame_idx,
                        )
                    )

            # Build pass arrow list with age fractions and prune expired.
            _active_pass_arrows = [
                a
                for a in _active_pass_arrows
                if frame_data.frame_idx - a[3] <= _pass_arrow_duration
            ]
            minimap_pass_arrows = [
                (a[0], a[1], a[2], (frame_data.frame_idx - a[3]) / _pass_arrow_duration)
                for a in _active_pass_arrows
            ]

            _draw_possession_indicator(yolo_frame, tracks, possessor_id)
            yolo_writer.write(yolo_frame)

            tactical_frame = frame_data.img.copy()
            _draw_tracks(
                tactical_frame,
                tracks,
                include_team=enable_team_assignment,
                team_colored=enable_team_assignment,
            )
            _draw_possession_indicator(tactical_frame, tracks, possessor_id)
            if _should_render_keypoint_debug(
                frame_idx=frame_data.frame_idx,
                enabled=debug_keypoints,
                start_frame=debug_keypoints_start_frame,
                end_frame=debug_keypoints_end_frame,
                last_n_frames=debug_keypoints_last_n,
                total_output_frames=total_output_frames,
            ):
                _draw_pitch_keypoint_debug(
                    tactical_frame,
                    keypoints_xy=keypoints_xy,
                    keypoints_conf=keypoints_conf,
                    conf_threshold=homography_keypoint_conf_threshold,
                )
            split_frame = minimap_renderer.render_split_panel(
                tactical_frame,
                tracks,
                panel_height=radar_panel_height,
                status_text=_status_text(h_status),
                ball_speed_kmh=ball_speed_kmh,
                possessor_id=possessor_id,
                pass_arrows=minimap_pass_arrows if minimap_pass_arrows else None,
            )
            if team_possession_tracker is not None:
                _draw_possession_bar(
                    split_frame[height:],  # panel portion only
                    radar_panel_height,
                    width,
                    team_possession_tracker.percentages(),
                )
            if pass_detector is not None:
                _draw_pass_counts(
                    split_frame[height:],  # panel portion only
                    radar_panel_height,
                    width,
                    pass_detector.summary(),
                )
                _draw_turnover_counts(
                    split_frame[height:],
                    radar_panel_height,
                    width,
                    pass_detector.turnover_summary(),
                )
            tactical_writer.write(split_frame)
    finally:
        yolo_writer.release()
        tactical_writer.release()
        if csv_writer is not None:
            csv_writer.close()

    if team_possession_tracker is not None:
        pcts = team_possession_tracker.percentages()
        parts = [f"Team {tid}: {pct:.1f}%" for tid, pct in sorted(pcts.items())]
        print(f"Possession: {', '.join(parts) if parts else 'N/A'}")

    if pass_detector is not None:
        summary = pass_detector.summary()
        parts = [f"Team {tid}: {s['completed']}" for tid, s in sorted(summary.items())]
        print(f"Passes: {', '.join(parts) if parts else 'N/A'}")

        t_summary = pass_detector.turnover_summary()
        t_parts = [
            f"Team {tid}: {s['lost']} lost" for tid, s in sorted(t_summary.items())
        ]
        print(f"Turnovers: {', '.join(t_parts) if t_parts else 'N/A'}")

    if _want_heatmaps and accum_positions:
        from src.io.generate_player_heatmap import generate_heatmaps

        heatmap_positions = {
            tid: [(x, y) for _, x, y in samples]
            for tid, samples in accum_positions.items()
        }
        n = generate_heatmaps(
            heatmap_positions,
            accum_metadata,
            pitch_cfg,
            heatmap_output_dir,
            track_ids=heatmap_track_ids,
            top_n=heatmap_top_n,
        )
        if n > 0:
            print(f"Generated {n} heatmap(s) in {heatmap_output_dir}")

    if _want_trajectories and accum_positions:
        from src.io.generate_pitch_trajectories import generate_trajectories

        n = generate_trajectories(
            accum_positions,
            accum_metadata,
            pitch_cfg,
            trajectory_output_dir,
            views=trajectory_views,
            top_k=trajectory_top_k,
            track_ids=trajectory_track_ids,
        )
        if n > 0:
            print(f"Generated {n} trajectory image(s) in {trajectory_output_dir}")

    if _want_report and accum_positions:
        from src.io.generate_match_report import generate_report

        player_csv, team_csv = generate_report(
            accum_positions,
            accum_metadata,
            fps,
            report_output_dir,
        )
        print(f"Player report: {player_csv}")
        print(f"Team report: {team_csv}")

    if _want_pass_network and pass_detector is not None and pass_detector.passes:
        from src.io.generate_pass_network import generate_pass_network

        n = generate_pass_network(
            passes=pass_detector.passes,
            accum_positions=accum_positions,
            accum_metadata=accum_metadata,
            pitch_cfg=pitch_cfg,
            output_dir=pass_network_output_dir,
        )
        if n > 0:
            print(f"Generated {n} pass network image(s) in {pass_network_output_dir}")

    return processed_frames
