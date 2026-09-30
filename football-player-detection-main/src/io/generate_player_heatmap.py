import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.core.pitch import SoccerPitchConfiguration


def world_to_canvas(
    x_cm: float,
    y_cm: float,
    pitch_length_cm: float,
    pitch_width_cm: float,
    width_px: int,
    height_px: int,
) -> Tuple[int, int]:
    """Map pitch-space coordinates (cm) to output image pixel coordinates."""
    x_px = int(np.clip((x_cm / pitch_length_cm) * (width_px - 1), 0, width_px - 1))
    y_px = int(np.clip((y_cm / pitch_width_cm) * (height_px - 1), 0, height_px - 1))
    return x_px, y_px


def draw_pitch(
    width_px: int,
    height_px: int,
    cfg: SoccerPitchConfiguration,
) -> np.ndarray:
    """Create a blank pitch image used as the background for the heatmap."""
    canvas = np.full((height_px, width_px, 3), (38, 130, 56), dtype=np.uint8)
    line_color = (240, 240, 240)
    length = float(cfg.length)
    width = float(cfg.width)

    def pt(x_cm: float, y_cm: float) -> Tuple[int, int]:
        return world_to_canvas(x_cm, y_cm, length, width, width_px, height_px)

    # Outer boundary.
    cv2.rectangle(
        canvas, pt(0.0, 0.0), pt(length, width), line_color, 2, lineType=cv2.LINE_AA
    )

    # Halfway line.
    cv2.line(
        canvas,
        pt(length / 2.0, 0.0),
        pt(length / 2.0, width),
        line_color,
        2,
        lineType=cv2.LINE_AA,
    )

    # Center circle and spot.
    center = pt(length / 2.0, width / 2.0)
    radius_px = max(int((cfg.centre_circle_radius / length) * (width_px - 1)), 1)
    cv2.circle(canvas, center, radius_px, line_color, 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, 2, line_color, -1, lineType=cv2.LINE_AA)

    # Penalty boxes.
    pb_top = (width - cfg.penalty_box_width) / 2.0
    pb_bot = (width + cfg.penalty_box_width) / 2.0
    cv2.rectangle(
        canvas,
        pt(0.0, pb_top),
        pt(cfg.penalty_box_length, pb_bot),
        line_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        pt(length - cfg.penalty_box_length, pb_top),
        pt(length, pb_bot),
        line_color,
        1,
        lineType=cv2.LINE_AA,
    )

    # Goal boxes.
    gb_top = (width - cfg.goal_box_width) / 2.0
    gb_bot = (width + cfg.goal_box_width) / 2.0
    cv2.rectangle(
        canvas,
        pt(0.0, gb_top),
        pt(cfg.goal_box_length, gb_bot),
        line_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        pt(length - cfg.goal_box_length, gb_top),
        pt(length, gb_bot),
        line_color,
        1,
        lineType=cv2.LINE_AA,
    )

    # Penalty spots.
    cv2.circle(
        canvas,
        pt(cfg.penalty_spot_distance, width / 2.0),
        2,
        line_color,
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        pt(length - cfg.penalty_spot_distance, width / 2.0),
        2,
        line_color,
        -1,
        lineType=cv2.LINE_AA,
    )

    return canvas


def build_density_map(
    xs_cm: np.ndarray,
    ys_cm: np.ndarray,
    pitch_length_cm: float,
    pitch_width_cm: float,
    width_px: int,
    height_px: int,
    blur_sigma_px: float,
) -> np.ndarray:
    """Convert player samples into a normalized 2D density map (0..1)."""
    density = np.zeros((height_px, width_px), dtype=np.float32)

    for x_cm, y_cm in zip(xs_cm, ys_cm):
        x_px, y_px = world_to_canvas(
            float(x_cm),
            float(y_cm),
            pitch_length_cm=pitch_length_cm,
            pitch_width_cm=pitch_width_cm,
            width_px=width_px,
            height_px=height_px,
        )
        density[y_px, x_px] += 1.0

    if blur_sigma_px > 0:
        density = cv2.GaussianBlur(
            density, ksize=(0, 0), sigmaX=blur_sigma_px, sigmaY=blur_sigma_px
        )

    max_val = float(density.max())
    if max_val > 0:
        density = density / max_val
    return density


def _render_single_heatmap(
    positions: list[tuple[float, float]],
    pitch_cfg: SoccerPitchConfiguration,
    output_path: Path,
    track_id: int,
    team_id: int | None,
    width_px: int = 1000,
    height_px: int = 648,
    blur_sigma_px: float = 14.0,
) -> None:
    """Render and save a single heatmap PNG from in-memory positions (cm)."""
    xs = np.array([p[0] for p in positions], dtype=np.float32)
    ys = np.array([p[1] for p in positions], dtype=np.float32)

    base = draw_pitch(width_px, height_px, pitch_cfg)
    density = build_density_map(
        xs_cm=xs,
        ys_cm=ys,
        pitch_length_cm=float(pitch_cfg.length),
        pitch_width_cm=float(pitch_cfg.width),
        width_px=width_px,
        height_px=height_px,
        blur_sigma_px=blur_sigma_px,
    )

    heat_u8 = (density * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    alpha = 0.62
    mask = heat_u8 > 0
    out = base.copy()
    out[mask] = cv2.addWeighted(base[mask], 1.0 - alpha, heat_color[mask], alpha, 0)

    team_str = str(team_id) if team_id is not None else "?"
    title = (
        f"Player Heatmap | track_id={track_id} team={team_str} samples={len(positions)}"
    )
    cv2.putText(
        out,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def generate_heatmaps(
    track_positions: dict[int, list[tuple[float, float]]],
    track_metadata: dict[int, tuple[str, int | None]],
    pitch_cfg: SoccerPitchConfiguration,
    output_dir: Path,
    *,
    track_ids: list[int] | None = None,
    top_n: int | None = None,
    min_samples: int = 30,
    width_px: int = 1000,
    height_px: int = 648,
    blur_sigma_px: float = 14.0,
) -> int:
    """
    Generate heatmap PNGs for selected tracks from in-memory position data.

    Returns number of heatmaps generated.
    """
    # Filter to player/goalkeeper tracks with enough samples.
    eligible: dict[int, list[tuple[float, float]]] = {}
    for tid, positions in track_positions.items():
        cls_name, _ = track_metadata.get(tid, ("", None))
        if cls_name not in {"Player", "Goalkeeper"}:
            continue
        if len(positions) < min_samples:
            continue
        eligible[tid] = positions

    # Determine which tracks to render.
    selected: set[int] = set()
    if track_ids:
        selected.update(tid for tid in track_ids if tid in eligible)
    if top_n and top_n > 0:
        by_count = sorted(
            eligible.keys(), key=lambda tid: len(eligible[tid]), reverse=True
        )
        selected.update(by_count[:top_n])

    if not selected:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for tid in sorted(selected):
        _, team_id = track_metadata.get(tid, ("", None))
        _render_single_heatmap(
            positions=eligible[tid],
            pitch_cfg=pitch_cfg,
            output_path=output_dir / f"heatmap_track{tid}.png",
            track_id=tid,
            team_id=team_id,
            width_px=width_px,
            height_px=height_px,
            blur_sigma_px=blur_sigma_px,
        )
        count += 1

    return count


def make_player_heatmap(
    tracks_csv: Path,
    output_path: Path,
    track_id: int | None = None,
    cls_name: str = "Player",
    width_px: int = 1000,
    height_px: int = 648,
    blur_sigma_px: float = 14.0,
) -> Tuple[int, int]:
    """
    Build and save one player heatmap image from a CSV file.

    CSV columns pitch_x_m / pitch_y_m are treated as meters and converted to cm.

    Returns:
        (track_id_used, num_samples_used)
    """
    import pandas as pd

    if not tracks_csv.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_csv}")

    df = pd.read_csv(tracks_csv)
    required_cols = {
        "track_id",
        "cls_name",
        "pitch_point_valid",
        "pitch_x_m",
        "pitch_y_m",
        "frame",
        "team_id",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {tracks_csv}: {missing}")

    if track_id is None:
        d = df[(df["cls_name"] == cls_name) & (df["pitch_point_valid"] == 1)]
        if d.empty:
            raise ValueError(f"No valid projected points found for class '{cls_name}'.")
        counts = d.groupby("track_id").size().sort_values(ascending=False)
        track_id = int(counts.index[0])

    d = df[
        (df["track_id"] == track_id)
        & (df["cls_name"] == cls_name)
        & (df["pitch_point_valid"] == 1)
    ].copy()
    d = d.sort_values("frame")
    if d.empty:
        raise ValueError(
            f"No valid projected points for track_id={track_id} and cls_name={cls_name}."
        )

    cfg = SoccerPitchConfiguration()
    # CSV values are in meters; convert to cm.
    xs_cm = d["pitch_x_m"].to_numpy(dtype=np.float32) * 100.0
    ys_cm = d["pitch_y_m"].to_numpy(dtype=np.float32) * 100.0

    base = draw_pitch(width_px, height_px, cfg)
    density = build_density_map(
        xs_cm=xs_cm,
        ys_cm=ys_cm,
        pitch_length_cm=float(cfg.length),
        pitch_width_cm=float(cfg.width),
        width_px=width_px,
        height_px=height_px,
        blur_sigma_px=blur_sigma_px,
    )

    heat_u8 = (density * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    alpha = 0.62
    mask = heat_u8 > 0
    out = base.copy()
    out[mask] = cv2.addWeighted(base[mask], 1.0 - alpha, heat_color[mask], alpha, 0)

    team_mode = (
        int(d["team_id"].mode().iloc[0]) if not d["team_id"].mode().empty else -1
    )
    title = f"Player Heatmap | track_id={track_id} team={team_mode} samples={len(d)}"
    cv2.putText(
        out,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), out)
    if not ok:
        raise RuntimeError(f"Failed to write heatmap image to: {output_path}")

    return track_id, len(d)


def main() -> None:
    """CLI entrypoint for generating per-player heatmaps from tracks.csv."""
    parser = argparse.ArgumentParser(
        description="Generate per-player heatmap from projected pitch coordinates."
    )

    parser.add_argument(
        "--tracks",
        default="outputs/video_tracking/botsort_test/tracks.csv",
        help="Path to tracks CSV (default: outputs/video_tracking/botsort_test/tracks.csv)",
    )

    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Track ID to render. If omitted, auto-selects the player with most valid points.",
    )
    parser.add_argument(
        "--class-name",
        default="Player",
        help="Class filter for heatmap generation (default: Player)",
    )

    parser.add_argument(
        "--output",
        default="outputs/video_tracking/botsort_test/heatmaps/player_heatmap.png",
        help="Output image path",
    )
    parser.add_argument(
        "--width", type=int, default=1000, help="Output width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=648, help="Output height in pixels"
    )
    parser.add_argument(
        "--blur-sigma", type=float, default=14.0, help="Gaussian blur sigma in pixels"
    )
    args = parser.parse_args()

    track_id, num_samples = make_player_heatmap(
        tracks_csv=Path(args.tracks),
        output_path=Path(args.output),
        track_id=args.track_id,
        cls_name=args.class_name,
        width_px=args.width,
        height_px=args.height,
        blur_sigma_px=args.blur_sigma,
    )
    print(f"Saved heatmap to: {args.output}")
    print(f"track_id={track_id}, class={args.class_name}, samples={num_samples}")


if __name__ == "__main__":
    main()
