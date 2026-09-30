import argparse
from pathlib import Path

import cv2
import numpy as np

from src.core.pitch import SoccerPitchConfiguration
from src.io.generate_player_heatmap import draw_pitch, world_to_canvas


def _draw_polylines(
    base: np.ndarray,
    tracks: list[dict],
    color_fn,
    line_thickness: int,
) -> np.ndarray:
    out = base.copy()
    for r in tracks:
        pts = np.array(r["polyline"], dtype=np.int32)
        if len(pts) < 2:
            continue
        color = color_fn(r)
        cv2.polylines(
            out,
            [pts],
            isClosed=False,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(out, tuple(pts[-1]), 2, color, -1, lineType=cv2.LINE_AA)
    return out


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _draw_header(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        lineType=cv2.LINE_AA,
    )
    return out


def _build_track_records(
    track_positions: dict[int, list[tuple[int, float, float]]],
    track_metadata: dict[int, tuple[str, int | None]],
    pitch_cfg: SoccerPitchConfiguration,
    width_px: int,
    height_px: int,
    min_points: int,
    max_gap_frames: int,
) -> list[dict]:
    """Build renderable track records from in-memory position data."""
    pitch_length_cm = float(pitch_cfg.length)
    pitch_width_cm = float(pitch_cfg.width)
    records = []

    for track_id, samples in track_positions.items():
        cls_name, team_id = track_metadata.get(track_id, ("", None))
        if cls_name not in {"Player", "Goalkeeper"}:
            continue
        if len(samples) < min_points:
            continue

        # Sort by frame index.
        samples_sorted = sorted(samples, key=lambda s: s[0])
        frames = [s[0] for s in samples_sorted]
        xs = [s[1] for s in samples_sorted]
        ys = [s[2] for s in samples_sorted]

        # Compute distance (cm) from consecutive projected points.
        dist_cm = 0.0
        if len(xs) >= 2:
            for i in range(1, len(xs)):
                gap = frames[i] - frames[i - 1]
                if 1 <= gap <= max_gap_frames:
                    dx = xs[i] - xs[i - 1]
                    dy = ys[i] - ys[i - 1]
                    dist_cm += (dx * dx + dy * dy) ** 0.5

        pts = [
            world_to_canvas(x, y, pitch_length_cm, pitch_width_cm, width_px, height_px)
            for x, y in zip(xs, ys)
        ]

        # Deduplicate consecutive identical pixels.
        dedup: list[tuple[int, int]] = []
        for p in pts:
            if not dedup or dedup[-1] != p:
                dedup.append(p)

        if len(dedup) < min_points:
            continue

        records.append(
            {
                "track_id": track_id,
                "team_id": team_id if team_id is not None else -1,
                "cls_name": cls_name,
                "distance_cm": dist_cm,
                "num_points": len(dedup),
                "polyline": dedup,
            }
        )

    return records


def _render_all_players_view(
    base: np.ndarray,
    records: list[dict],
    output_dir: Path,
    line_thickness: int,
) -> Path:
    title = f"All Players Trajectories | tracks={len(records)}"
    out_path = output_dir / "all_players_trajectories.png"

    def color_fn(row):
        if row["team_id"] == 0:
            return (0, 0, 255)
        if row["team_id"] == 1:
            return (255, 0, 0)
        return (180, 180, 180)

    img = _draw_polylines(base, records, color_fn, line_thickness)
    img = _draw_header(img, title)
    _save_image(out_path, img)
    return out_path


def _render_team_view(
    base: np.ndarray,
    team_records: list[dict],
    output_dir: Path,
    line_thickness: int,
    team_id: int,
) -> Path:
    title = f"Team Trajectories | team={team_id} tracks={len(team_records)}"
    out_path = output_dir / f"team_{team_id}_trajectories.png"

    img = _draw_polylines(base, team_records, lambda _r: (0, 220, 255), line_thickness)
    img = _draw_header(img, title)
    _save_image(out_path, img)
    return out_path


def _render_topk_view(
    base: np.ndarray,
    records: list[dict],
    output_dir: Path,
    line_thickness: int,
    top_k: int,
) -> Path:
    top = sorted(records, key=lambda r: r["distance_cm"], reverse=True)[:top_k]

    palette = [
        (0, 255, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (80, 80, 255),
    ]

    def color_fn(row):
        for i, r in enumerate(top):
            if r["track_id"] == row["track_id"]:
                return palette[i % len(palette)]
        return (180, 180, 180)

    img = _draw_polylines(base, top, color_fn, line_thickness)
    img = _draw_header(img, f"Top-{top_k} Distance Trajectories")

    y = 56
    for i, r in enumerate(top):
        color = palette[i % len(palette)]
        dist_m = r["distance_cm"] / 100.0
        label = f"{i + 1}) track={r['track_id']} team={r['team_id']} dist={dist_m:.1f}m"
        cv2.putText(
            img,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        y += 24

    out_path = output_dir / f"top_{top_k}_trajectories.png"
    _save_image(out_path, img)
    return out_path


def _render_player_view(
    base: np.ndarray,
    record: dict,
    output_dir: Path,
    line_thickness: int,
) -> Path:
    img = _draw_polylines(
        base, [record], lambda _r: (0, 255, 255), max(2, line_thickness)
    )
    dist_m = record["distance_cm"] / 100.0
    title = (
        f"Player Trajectory | track={record['track_id']} "
        f"team={record['team_id']} dist={dist_m:.1f}m points={record['num_points']}"
    )
    img = _draw_header(img, title)
    out_path = output_dir / f"player_{record['track_id']}_trajectory.png"
    _save_image(out_path, img)
    return out_path


def generate_trajectories(
    track_positions: dict[int, list[tuple[int, float, float]]],
    track_metadata: dict[int, tuple[str, int | None]],
    pitch_cfg: SoccerPitchConfiguration,
    output_dir: Path,
    *,
    views: set[str] | None = None,
    top_k: int = 3,
    track_ids: list[int] | None = None,
    min_points: int = 8,
    max_gap_frames: int = 5,
    width_px: int = 1000,
    height_px: int = 648,
    line_thickness: int = 1,
) -> int:
    """
    Generate trajectory PNGs from in-memory position data.

    Args:
        track_positions: track_id -> [(frame_idx, x_cm, y_cm), ...]
        track_metadata: track_id -> (cls_name, team_id)
        views: subset of {"allplayers", "team", "topk", "player"}.
               Defaults to {"allplayers", "team", "topk"}.
        track_ids: if provided with "player" view, render those specific tracks.
        top_k: number of top-distance tracks for "topk" view.

    Returns:
        Number of images generated.
    """
    if views is None:
        views = {"allplayers", "team", "topk"}

    records = _build_track_records(
        track_positions,
        track_metadata,
        pitch_cfg,
        width_px=width_px,
        height_px=height_px,
        min_points=min_points,
        max_gap_frames=max_gap_frames,
    )

    if not records:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    base = draw_pitch(width_px, height_px, pitch_cfg)
    count = 0

    if "allplayers" in views:
        _render_all_players_view(base, records, output_dir, line_thickness)
        count += 1

    if "team" in views:
        team_ids_present = sorted({r["team_id"] for r in records if r["team_id"] >= 0})
        for tid in team_ids_present:
            team_recs = [r for r in records if r["team_id"] == tid]
            if team_recs:
                _render_team_view(
                    base, team_recs, output_dir, max(2, line_thickness), tid
                )
                count += 1

    if "topk" in views:
        _render_topk_view(base, records, output_dir, max(2, line_thickness), top_k)
        count += 1

    if "player" in views and track_ids:
        records_by_id = {r["track_id"]: r for r in records}
        for tid in track_ids:
            if tid in records_by_id:
                _render_player_view(
                    base, records_by_id[tid], output_dir, line_thickness
                )
                count += 1

    return count


def main() -> None:
    """CLI entrypoint for generating trajectory views from tracks.csv."""
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Generate pitch trajectory views from tracks.csv"
    )
    parser.add_argument("--tracks", required=True, help="Path to tracks.csv")
    parser.add_argument(
        "--views",
        default="allplayers,team,topk,player",
        help="Comma-separated views: allplayers,team,topk,player",
    )
    parser.add_argument(
        "--classes",
        default="Player,Goalkeeper",
        help="Comma-separated classes to include (default: Player,Goalkeeper)",
    )
    parser.add_argument("--output-dir", default=None, help="Output folder")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Top-k for topk view (default: 3)"
    )
    parser.add_argument(
        "--player-track-id",
        type=int,
        action="append",
        default=None,
        help="Track ID for player view. Repeatable.",
    )
    parser.add_argument("--width", type=int, default=1000, help="Output width px")
    parser.add_argument("--height", type=int, default=648, help="Output height px")
    parser.add_argument(
        "--line-thickness", type=int, default=1, help="Polyline thickness"
    )
    parser.add_argument(
        "--min-points", type=int, default=8, help="Min polyline points per track"
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=1,
        help="Max frame gap for distance calculation",
    )
    args = parser.parse_args()

    tracks_path = Path(args.tracks)
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else tracks_path.parent / "trajectories"
    )

    views_set = {x.strip().lower() for x in args.views.split(",") if x.strip()}
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]

    df = pd.read_csv(tracks_path)
    required = ["frame", "track_id", "cls_name", "team_id", "pitch_x_m", "pitch_y_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["cls_name"].isin(classes)].copy()
    if "pitch_point_valid" in df.columns:
        df = df[df["pitch_point_valid"].fillna(0).astype(int) == 1]
    df = df.dropna(subset=["pitch_x_m", "pitch_y_m", "frame", "track_id"])

    # Build in-memory dicts (CSV is in meters, convert to cm).
    track_positions: dict[int, list[tuple[int, float, float]]] = {}
    track_metadata: dict[int, tuple[str, int | None]] = {}
    for _, row in df.iterrows():
        tid = int(row["track_id"])
        frame_idx = int(row["frame"])
        x_cm = float(row["pitch_x_m"]) * 100.0
        y_cm = float(row["pitch_y_m"]) * 100.0
        track_positions.setdefault(tid, []).append((frame_idx, x_cm, y_cm))
        team_id = int(row["team_id"]) if row["team_id"] >= 0 else None
        track_metadata[tid] = (str(row["cls_name"]), team_id)

    cfg = SoccerPitchConfiguration()

    # Auto-select player track IDs if "player" view requested without explicit IDs.
    player_ids = args.player_track_id
    if "player" in views_set and not player_ids:
        # Pick the top-distance player.
        records = _build_track_records(
            track_positions,
            track_metadata,
            cfg,
            width_px=args.width,
            height_px=args.height,
            min_points=args.min_points,
            max_gap_frames=args.max_gap_frames,
        )
        if records:
            best = max(records, key=lambda r: r["distance_cm"])
            player_ids = [best["track_id"]]

    n = generate_trajectories(
        track_positions,
        track_metadata,
        cfg,
        output_dir,
        views=views_set,
        top_k=args.top_k,
        track_ids=player_ids,
        min_points=args.min_points,
        max_gap_frames=args.max_gap_frames,
        width_px=args.width,
        height_px=args.height,
        line_thickness=args.line_thickness,
    )
    print(f"Generated {n} trajectory image(s) in {output_dir}")


if __name__ == "__main__":
    main()
