import argparse
import inspect
from pathlib import Path
from typing import Iterator

import numpy as np
from ultralytics import YOLO

from src.app.app import run_video_pipeline
from src.core.pitch import SoccerPitchConfiguration
from src.vision.pitch_keypoint_detector import PitchKeypointDetector
from src.vision.tracker import Tracker

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_BALL_TRACKER_WEIGHTS = "models/ball_tracking_1280_e300.pt"
DEFAULT_PLAYER_TRACKER_WEIGHTS = "models/best_players_gk_1280_s_e300.pt"
DEFAULT_PITCH_WEIGHTS = "models/pitch_kpts32_y8s_640_e500_AO.pt"
DEFAULT_TRACKER_CFG = "configs/botsort.yaml"


def resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _import_supervision():
    try:
        import supervision as sv
    except ImportError as exc:
        raise ImportError(
            "supervision is required for run_pitch_detection/run_ball_detection. "
            "Install it with: pip install supervision"
        ) from exc
    return sv


def _build_inference_slicer(sv, callback):
    sig = inspect.signature(sv.InferenceSlicer)
    kwargs = {
        "callback": callback,
        "slice_wh": (640, 640),
    }
    if "overlap_filter_strategy" in sig.parameters:
        kwargs["overlap_filter_strategy"] = sv.OverlapFilter.NONE
    else:
        kwargs["overlap_filter"] = sv.OverlapFilter.NONE
    return sv.InferenceSlicer(**kwargs)


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch keypoint detection and yield frames annotated with vertex labels.
    """
    sv = _import_supervision()
    config = SoccerPitchConfiguration()
    vertex_label_annotator = sv.VertexLabelAnnotator(
        color=[sv.Color.from_hex(color) for color in config.colors],
        text_color=sv.Color.from_hex("#FFFFFF"),
        border_radius=5,
        text_thickness=1,
        text_scale=0.5,
        text_padding=5,
    )

    pitch_detection_model = YOLO(DEFAULT_PITCH_WEIGHTS).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated_frame = vertex_label_annotator.annotate(
            frame.copy(), keypoints, config.labels
        )
        yield annotated_frame


def run_ball_detection(
    source_video_path: str, device: str = "mps"
) -> Iterator[np.ndarray]:
    """
    Run ball detection/tracking and yield annotated frames.
    """
    sv = _import_supervision()
    from src.vision.ball import BallAnnotator, BallTracker

    model = YOLO(DEFAULT_BALL_TRACKER_WEIGHTS).to(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def fallback(image_slice: np.ndarray) -> "sv.Detections":
        result = model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = _build_inference_slicer(sv, fallback)

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = ball_annotator.annotate(frame.copy(), detections)
        yield annotated_frame


def build_ball_overlay_fn(ball_weights: str, device: str = "mps"):
    """
    Build a per-frame ball overlay callback using the separate ball model.
    """
    sv = _import_supervision()
    from src.vision.ball import BallAnnotator, BallTracker

    model = YOLO(ball_weights).to(device)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def fallback(image_slice: np.ndarray) -> "sv.Detections":
        result = model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = _build_inference_slicer(sv, fallback)

    def overlay_fn(
        frame: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float] | None]:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated = ball_annotator.annotate(frame, detections)
        if len(detections) == 0:
            return annotated, None
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        return annotated, (float(xy[0][0]), float(xy[0][1]))

    return overlay_fn


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two videos: "
            "1) YOLO tracks for players/goalkeepers/referees, "
            "2) team assignment + homography minimap overlay."
        )
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "outputs/video_tracking"),
        help="Base output directory.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run directory name under output-dir (default: input video stem).",
    )
    parser.add_argument(
        "--yolo-video",
        default=None,
        help="Optional explicit output path for YOLO-tracking video.",
    )
    parser.add_argument(
        "--team-video",
        default=None,
        help="Optional explicit output path for team+homography video.",
    )
    parser.add_argument(
        "--tracker-weights",
        default=str(project_root / DEFAULT_PLAYER_TRACKER_WEIGHTS),
        help="Path to tracker model weights.",
    )
    parser.add_argument(
        "--ball-weights",
        default=str(project_root / DEFAULT_BALL_TRACKER_WEIGHTS),
        help="Path to separate ball model weights.",
    )
    parser.add_argument(
        "--pitch-weights",
        default=str(project_root / DEFAULT_PITCH_WEIGHTS),
        help="Path to pitch keypoint model weights.",
    )
    parser.add_argument(
        "--tracker-cfg",
        default=str(project_root / DEFAULT_TRACKER_CFG),
        help="Path to tracker config file.",
    )
    parser.add_argument("--device", default="mps", help="Device for tracker model.")
    parser.add_argument("--ball-device", default="mps", help="Device for ball model.")
    parser.add_argument(
        "--pitch-device", default="cpu", help="Device for pitch keypoint model."
    )
    parser.add_argument(
        "--vid-stride", type=int, default=1, help="Process every Nth frame."
    )
    parser.add_argument(
        "--homography-keypoint-conf-threshold",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold for homography.",
    )
    parser.add_argument(
        "--homography-min-points",
        type=int,
        default=6,
        help="Minimum keypoints required to solve homography.",
    )
    parser.add_argument(
        "--homography-max-stale-frames",
        type=int,
        default=60,
        help="Maximum stale frames allowed for homography fallback.",
    )
    parser.add_argument(
        "--disable-team-assignment",
        action="store_true",
        help="Disable team assignment (second video still includes homography minimap).",
    )
    parser.add_argument(
        "--disable-possession",
        action="store_true",
        help="Disable ball possession detection indicator.",
    )
    parser.add_argument(
        "--debug-keypoints",
        action="store_true",
        help=(
            "Draw pitch keypoint debug overlay on the team/homography video. "
            "By default this renders on the last --debug-keypoints-last-n frames."
        ),
    )
    parser.add_argument(
        "--debug-homography-diag",
        action="store_true",
        help="Print compact per-frame keypoint and homography diagnostics.",
    )
    parser.add_argument(
        "--debug-keypoints-last-n",
        type=int,
        default=100,
        help="Number of final output frames to render keypoint debug on (default: 100).",
    )
    parser.add_argument(
        "--debug-keypoints-start-frame",
        type=int,
        default=None,
        help="Optional start frame (1-based output frame index) for keypoint debug rendering.",
    )
    parser.add_argument(
        "--debug-keypoints-end-frame",
        type=int,
        default=None,
        help="Optional end frame (1-based output frame index) for keypoint debug rendering.",
    )
    parser.add_argument(
        "--heatmaps",
        type=int,
        default=0,
        help="Generate heatmaps for top N players by sample count. 0 = off (default: 0).",
    )
    parser.add_argument(
        "--heatmap-track",
        type=int,
        action="append",
        default=None,
        help="Generate heatmap for a specific track ID. Repeatable.",
    )
    parser.add_argument(
        "--trajectories",
        action="store_true",
        help="Generate trajectory images (allplayers, team, topk views).",
    )
    parser.add_argument(
        "--trajectory-top-k",
        type=int,
        default=3,
        help="Top-k players by distance for trajectory topk view (default: 3).",
    )
    parser.add_argument(
        "--trajectory-track",
        type=int,
        action="append",
        default=None,
        help="Generate individual trajectory for a specific track ID. Repeatable.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate player/team summary CSVs (player_summary.csv, team_summary.csv).",
    )
    parser.add_argument(
        "--pass-network",
        action="store_true",
        help="Generate per-team pass network PNG images.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write per-frame tracks.csv with all detections and homography status.",
    )
    return parser


def main() -> None:
    args = build_parser(PROJECT_ROOT).parse_args()

    if args.vid_stride < 1:
        raise ValueError("--vid-stride must be >= 1")
    if args.homography_min_points < 4:
        raise ValueError("--homography-min-points must be >= 4")
    if args.homography_max_stale_frames < 0:
        raise ValueError("--homography-max-stale-frames must be >= 0")
    if args.debug_keypoints_last_n < 0:
        raise ValueError("--debug-keypoints-last-n must be >= 0")
    if (
        args.debug_keypoints_start_frame is not None
        and args.debug_keypoints_start_frame < 1
    ):
        raise ValueError("--debug-keypoints-start-frame must be >= 1")
    if (
        args.debug_keypoints_end_frame is not None
        and args.debug_keypoints_end_frame < 1
    ):
        raise ValueError("--debug-keypoints-end-frame must be >= 1")
    if (
        args.debug_keypoints_start_frame is not None
        and args.debug_keypoints_end_frame is not None
        and args.debug_keypoints_start_frame > args.debug_keypoints_end_frame
    ):
        raise ValueError(
            "--debug-keypoints-start-frame must be <= --debug-keypoints-end-frame"
        )

    video_path = resolve_path(args.video, PROJECT_ROOT)
    tracker_weights = resolve_path(args.tracker_weights, PROJECT_ROOT)
    ball_weights = resolve_path(args.ball_weights, PROJECT_ROOT)
    pitch_weights = resolve_path(args.pitch_weights, PROJECT_ROOT)
    tracker_cfg = resolve_path(args.tracker_cfg, PROJECT_ROOT)

    require_file(video_path, "Input video")
    require_file(tracker_weights, "Tracker weights")
    require_file(ball_weights, "Ball weights")
    require_file(pitch_weights, "Pitch keypoint weights")
    require_file(tracker_cfg, "Tracker config")

    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    run_name = args.run_name or video_path.stem
    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    yolo_video_path = (
        resolve_path(args.yolo_video, PROJECT_ROOT)
        if args.yolo_video
        else run_output_dir / "yolo_players_referees_goalkeepers_ball.mp4"
    )
    team_video_path = (
        resolve_path(args.team_video, PROJECT_ROOT)
        if args.team_video
        else run_output_dir / "team_assignment_homography_minimap.mp4"
    )
    yolo_video_path.parent.mkdir(parents=True, exist_ok=True)
    team_video_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = Tracker(
        weights=str(tracker_weights),
        device=args.device,
        image_size=1280,
        tracker_cfg=str(tracker_cfg),
    )
    pitch_detector = PitchKeypointDetector(
        weights=str(pitch_weights),
        device=args.pitch_device,
        image_size=640,
        num_keypoints=32,
    )
    ball_overlay_fn = build_ball_overlay_fn(
        ball_weights=str(ball_weights),
        device=args.ball_device,
    )

    heatmap_dir = (
        run_output_dir / "heatmaps"
        if (args.heatmaps > 0 or args.heatmap_track)
        else None
    )

    trajectory_dir = (
        run_output_dir / "trajectories"
        if (args.trajectories or args.trajectory_track)
        else None
    )
    trajectory_views = None
    if trajectory_dir is not None:
        trajectory_views = {"allplayers", "team", "topk"}
        if args.trajectory_track:
            trajectory_views.add("player")

    report_dir = run_output_dir if args.report else None

    pass_network_dir = run_output_dir / "pass_network" if args.pass_network else None

    csv_path = run_output_dir / "tracks.csv" if args.csv else None

    processed_frames = run_video_pipeline(
        video_path=video_path,
        tracker=tracker,
        pitch_detector=pitch_detector,
        yolo_video_path=yolo_video_path,
        team_homography_video_path=team_video_path,
        vid_stride=args.vid_stride,
        homography_keypoint_conf_threshold=args.homography_keypoint_conf_threshold,
        homography_min_points=args.homography_min_points,
        homography_max_stale_frames=args.homography_max_stale_frames,
        enable_team_assignment=not args.disable_team_assignment,
        ball_overlay_fn=ball_overlay_fn,
        enable_possession=not args.disable_possession,
        debug_keypoints=args.debug_keypoints,
        debug_homography_diag=args.debug_homography_diag,
        debug_keypoints_last_n=args.debug_keypoints_last_n,
        debug_keypoints_start_frame=args.debug_keypoints_start_frame,
        debug_keypoints_end_frame=args.debug_keypoints_end_frame,
        heatmap_output_dir=heatmap_dir,
        heatmap_top_n=args.heatmaps,
        heatmap_track_ids=args.heatmap_track,
        trajectory_output_dir=trajectory_dir,
        trajectory_views=trajectory_views,
        trajectory_top_k=args.trajectory_top_k,
        trajectory_track_ids=args.trajectory_track,
        report_output_dir=report_dir,
        pass_network_output_dir=pass_network_dir,
        csv_output_path=csv_path,
    )

    print(f"Processed frames: {processed_frames}")
    print(f"YOLO video: {yolo_video_path}")
    print(f"Team+homography video: {team_video_path}")


if __name__ == "__main__":
    main()
