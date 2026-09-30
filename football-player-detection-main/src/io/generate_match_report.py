import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: Iterable[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _parse_classes(raw: str) -> List[str]:
    classes = [x.strip() for x in raw.split(",") if x.strip()]
    if not classes:
        raise ValueError("At least one class must be provided in --classes")
    return classes


def _pitch_valid_mask(df: pd.DataFrame) -> pd.Series:
    if "pitch_point_valid" in df.columns:
        return df["pitch_point_valid"].fillna(0).astype(int) == 1

    # Fallback for older CSVs without pitch_point_valid.
    x_ok = pd.to_numeric(df["pitch_x_m"], errors="coerce").notna()
    y_ok = pd.to_numeric(df["pitch_y_m"], errors="coerce").notna()
    return x_ok & y_ok & (df["pitch_x_m"] >= 0) & (df["pitch_y_m"] >= 0)


def _stable_team_id(track_df: pd.DataFrame) -> int:
    valid = track_df["team_id"][track_df["team_id"] >= 0]
    if valid.empty:
        return -1
    return int(valid.mode().iloc[0])


def _ema_1d(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _track_metrics(
    track_df: pd.DataFrame,
    fps: float,
    max_gap_frames: int,
    ema_alpha: float,
    max_step_m_per_frame: float,
    speed_cap_mps: Optional[float],
) -> Dict[str, float]:
    d = track_df.sort_values("frame").copy()
    frames_seen = int(d["frame"].nunique())
    seconds_seen = float(frames_seen / fps) if fps > 0 else 0.0

    valid = d[d["_pitch_valid"]].copy().sort_values("frame")

    total_distance_m = 0.0
    avg_speed_mps = 0.0
    max_speed_mps = 0.0
    max_speed_mps_capped = 0.0
    avg_abs_accel_mps2 = 0.0
    p95_abs_accel_mps2 = 0.0
    top_speed_time_s = np.nan
    total_elapsed_s = 0.0
    num_segments_used = 0
    num_segments_skipped_gap = 0
    num_segments_skipped_step = 0
    num_speed_clipped = 0

    if len(valid) >= 2:
        frames = valid["frame"].to_numpy(dtype=np.int64)
        xs = valid["pitch_x_m"].to_numpy(dtype=np.float64, copy=True)
        ys = valid["pitch_y_m"].to_numpy(dtype=np.float64, copy=True)

        # Smooth positions before derivatives to reduce frame-to-frame jitter.
        xs = _ema_1d(xs, alpha=ema_alpha)
        ys = _ema_1d(ys, alpha=ema_alpha)

        d_frames = np.diff(frames)
        dists = np.hypot(np.diff(xs), np.diff(ys))
        end_frames = frames[1:]

        # Only keep local continuity; never connect across long gaps.
        gap_ok = (d_frames >= 1) & (d_frames <= max_gap_frames)
        num_segments_skipped_gap = int((~gap_ok).sum())

        if np.any(gap_ok):
            d_frames_ok = d_frames[gap_ok]
            dists_ok = dists[gap_ok]
            end_frames_ok = end_frames[gap_ok]

            # Hard plausibility gate on per-frame displacement.
            if max_step_m_per_frame > 0:
                step_limit = max_step_m_per_frame * d_frames_ok
                step_ok = dists_ok <= step_limit
            else:
                step_ok = np.ones_like(dists_ok, dtype=bool)
            num_segments_skipped_step = int((~step_ok).sum())

            if np.any(step_ok):
                dt = d_frames_ok[step_ok] / fps
                dist = dists_ok[step_ok]
                seg_end_frames = end_frames_ok[step_ok]
                speeds = dist / dt

                num_segments_used = int(len(speeds))
                total_distance_m = float(dist.sum())
                total_elapsed_s = float(dt.sum())
                avg_speed_mps = (
                    float(total_distance_m / total_elapsed_s)
                    if total_elapsed_s > 0
                    else 0.0
                )

                i_max = int(np.argmax(speeds))
                max_speed_mps = float(speeds[i_max])
                top_speed_time_s = float(seg_end_frames[i_max] / fps)

                if speed_cap_mps is not None and speed_cap_mps > 0:
                    speeds_for_accel = np.minimum(speeds, speed_cap_mps)
                    num_speed_clipped = int((speeds > speed_cap_mps).sum())
                    max_speed_mps_capped = float(speeds_for_accel.max())
                else:
                    speeds_for_accel = speeds
                    max_speed_mps_capped = float(max_speed_mps)

                if len(speeds_for_accel) >= 2:
                    speed_times = seg_end_frames / fps
                    d_speed = np.diff(speeds_for_accel)
                    d_time = np.diff(speed_times)
                    accel_good = d_time > 0
                    if np.any(accel_good):
                        abs_accels = np.abs(d_speed[accel_good] / d_time[accel_good])
                        avg_abs_accel_mps2 = float(abs_accels.mean())
                        p95_abs_accel_mps2 = float(np.quantile(abs_accels, 0.95))
            else:
                max_speed_mps_capped = float(max_speed_mps)
        else:
            max_speed_mps_capped = float(max_speed_mps)

    return {
        "frames_seen": frames_seen,
        "seconds_seen": seconds_seen,
        "distance_m": float(total_distance_m),
        "avg_speed_mps": float(avg_speed_mps),
        "max_speed_mps": float(max_speed_mps),
        "max_speed_mps_capped": float(max_speed_mps_capped),
        "avg_abs_accel_mps2": float(avg_abs_accel_mps2),
        "p95_abs_accel_mps2": float(p95_abs_accel_mps2),
        "top_speed_time_s": float(top_speed_time_s)
        if np.isfinite(top_speed_time_s)
        else np.nan,
        "_movement_elapsed_s": float(total_elapsed_s),
        "num_segments_used": int(num_segments_used),
        "num_segments_skipped_gap": int(num_segments_skipped_gap),
        "num_segments_skipped_step": int(num_segments_skipped_step),
        "num_speed_clipped": int(num_speed_clipped),
    }


def build_player_summary(
    df: pd.DataFrame,
    fps: float,
    max_gap_frames: int,
    ema_alpha: float,
    max_step_m_per_frame: float,
    min_reliable_frames: int,
    speed_cap_mps: Optional[float],
) -> pd.DataFrame:
    rows = []
    for track_id, g in df.groupby("track_id", sort=True):
        metrics = _track_metrics(
            g,
            fps=fps,
            max_gap_frames=max_gap_frames,
            ema_alpha=ema_alpha,
            max_step_m_per_frame=max_step_m_per_frame,
            speed_cap_mps=speed_cap_mps,
        )
        rows.append(
            {
                "track_id": int(track_id),
                "team_id": _stable_team_id(g),
                "frames_seen": metrics["frames_seen"],
                "seconds_seen": metrics["seconds_seen"],
                "distance_m": metrics["distance_m"],
                "avg_speed_mps": metrics["avg_speed_mps"],
                "max_speed_mps": metrics["max_speed_mps"],
                "max_speed_mps_capped": metrics["max_speed_mps_capped"],
                "avg_accel_mps2": metrics["avg_abs_accel_mps2"],
                "p95_accel_mps2": metrics["p95_abs_accel_mps2"],
                "top_speed_time_s": metrics["top_speed_time_s"],
                "is_reliable": int(metrics["frames_seen"] >= min_reliable_frames),
                "num_segments_used": metrics["num_segments_used"],
                "num_segments_skipped_gap": metrics["num_segments_skipped_gap"],
                "num_segments_skipped_step": metrics["num_segments_skipped_step"],
                "num_speed_clipped": metrics["num_speed_clipped"],
                "_movement_elapsed_s": metrics["_movement_elapsed_s"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "team_id",
                "frames_seen",
                "seconds_seen",
                "distance_m",
                "avg_speed_mps",
                "max_speed_mps",
                "max_speed_mps_capped",
                "avg_accel_mps2",
                "p95_accel_mps2",
                "top_speed_time_s",
                "is_reliable",
                "num_segments_used",
                "num_segments_skipped_gap",
                "num_segments_skipped_step",
                "num_speed_clipped",
                "_movement_elapsed_s",
            ]
        )

    return out.sort_values(["team_id", "track_id"]).reset_index(drop=True)


def build_team_summary(
    filtered_df: pd.DataFrame,
    player_summary: pd.DataFrame,
    reliable_only: bool = True,
) -> pd.DataFrame:
    players = player_summary[player_summary["team_id"] >= 0].copy()
    if reliable_only:
        players = players[players["is_reliable"] == 1].copy()
    if players.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "total_distance_m",
                "mean_player_distance_m",
                "median_player_distance_m",
                "mean_team_speed_mps",
                "avg_active_players_per_frame",
                "players_used",
            ]
        )

    stable_team_map = players.set_index("track_id")["team_id"].to_dict()
    active = filtered_df[filtered_df["track_id"].isin(stable_team_map.keys())].copy()
    active["stable_team_id"] = active["track_id"].map(stable_team_map)

    min_frame = int(filtered_df["frame"].min())
    max_frame = int(filtered_df["frame"].max())
    full_frame_index = pd.RangeIndex(start=min_frame, stop=max_frame + 1)

    rows = []
    for team_id, g in players.groupby("team_id", sort=True):
        total_distance_m = float(g["distance_m"].sum())
        mean_player_distance_m = float(g["distance_m"].mean())
        median_player_distance_m = float(g["distance_m"].median())

        elapsed_s = float(g["_movement_elapsed_s"].sum())
        mean_team_speed_mps = (
            float(total_distance_m / elapsed_s) if elapsed_s > 0 else 0.0
        )

        team_active = active[active["stable_team_id"] == team_id]
        per_frame_counts = (
            team_active.groupby("frame")["track_id"]
            .nunique()
            .reindex(full_frame_index, fill_value=0)
        )
        avg_active_players_per_frame = (
            float(per_frame_counts.mean()) if len(per_frame_counts) else 0.0
        )

        rows.append(
            {
                "team_id": int(team_id),
                "total_distance_m": total_distance_m,
                "mean_player_distance_m": mean_player_distance_m,
                "median_player_distance_m": median_player_distance_m,
                "mean_team_speed_mps": mean_team_speed_mps,
                "avg_active_players_per_frame": avg_active_players_per_frame,
                "players_used": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)


def generate_report(
    track_positions: dict[int, list[tuple[int, float, float]]],
    track_metadata: dict[int, tuple[str, int | None]],
    fps: float,
    output_dir: Path,
    *,
    min_reliable_frames: int = 25,
    max_gap_frames: int = 5,
    ema_alpha: float = 0.3,
    max_step_m_per_frame: float = 2.0,
    speed_cap_mps: float | None = 12.0,
) -> tuple[Path, Path]:
    """Build player/team summary CSVs from in-memory accumulated positions.

    Parameters
    ----------
    track_positions : dict mapping track_id → list of (frame_idx, x_cm, y_cm)
    track_metadata  : dict mapping track_id → (cls_name, team_id)
    fps             : video frames per second
    output_dir      : directory to write CSVs into

    Returns
    -------
    (player_csv_path, team_csv_path)
    """
    rows = []
    for track_id, samples in track_positions.items():
        cls_name, team_id = track_metadata.get(track_id, ("Player", None))
        for frame_idx, x_cm, y_cm in samples:
            rows.append(
                {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "cls_name": cls_name,
                    "team_id": int(team_id) if team_id is not None else -1,
                    "pitch_x_m": x_cm / 100.0,
                    "pitch_y_m": y_cm / 100.0,
                    "_pitch_valid": True,
                }
            )

    if not rows:
        empty_player = pd.DataFrame(
            columns=[
                "track_id",
                "team_id",
                "frames_seen",
                "seconds_seen",
                "distance_m",
                "avg_speed_mps",
                "max_speed_mps",
            ]
        )
        empty_team = pd.DataFrame(
            columns=["team_id", "total_distance_m", "players_used"]
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        player_csv = output_dir / "player_summary.csv"
        team_csv = output_dir / "team_summary.csv"
        empty_player.to_csv(player_csv, index=False)
        empty_team.to_csv(team_csv, index=False)
        return player_csv, team_csv

    df = pd.DataFrame(rows)

    player_summary = build_player_summary(
        df,
        fps=fps,
        max_gap_frames=max_gap_frames,
        ema_alpha=ema_alpha,
        max_step_m_per_frame=max_step_m_per_frame,
        min_reliable_frames=min_reliable_frames,
        speed_cap_mps=speed_cap_mps,
    )

    team_summary = build_team_summary(
        filtered_df=df,
        player_summary=player_summary,
        reliable_only=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    player_csv = output_dir / "player_summary.csv"
    team_csv = output_dir / "team_summary.csv"

    player_summary.to_csv(
        player_csv,
        index=False,
        columns=[
            "track_id",
            "team_id",
            "frames_seen",
            "seconds_seen",
            "distance_m",
            "avg_speed_mps",
            "max_speed_mps",
            "max_speed_mps_capped",
            "avg_accel_mps2",
            "p95_accel_mps2",
            "top_speed_time_s",
            "is_reliable",
            "num_segments_used",
            "num_segments_skipped_gap",
            "num_segments_skipped_step",
            "num_speed_clipped",
        ],
    )
    team_summary.to_csv(
        team_csv,
        index=False,
        columns=[
            "team_id",
            "total_distance_m",
            "mean_player_distance_m",
            "median_player_distance_m",
            "mean_team_speed_mps",
            "avg_active_players_per_frame",
            "players_used",
        ],
    )

    print(f"Wrote player summary: {player_csv}")
    print(f"Wrote team summary: {team_csv}")
    print(f"Players: {len(player_summary)} | Teams: {len(team_summary)}")

    return player_csv, team_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate player/team match reports from tracks.csv"
    )
    parser.add_argument("--tracks", required=True, help="Path to tracks.csv")
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video FPS used for time-based metrics (default: 25.0)",
    )
    parser.add_argument(
        "--classes",
        default="Player,Goalkeeper",
        help="Comma-separated classes to include (default: Player,Goalkeeper)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSVs (default: same directory as tracks.csv)",
    )
    parser.add_argument(
        "--min-reliable-frames",
        type=int,
        default=25,
        help="Tracks below this frame count are flagged unreliable (default: 25)",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=1,
        help="Only use motion segments with frame gap <= this value (default: 1)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="EMA alpha for smoothing pitch positions before derivatives (default: 0.3)",
    )
    parser.add_argument(
        "--max-step-m-per-frame",
        type=float,
        default=2.0,
        help="Reject segments with displacement above this (meters/frame) (default: 2.0)",
    )
    parser.add_argument(
        "--speed-cap-mps",
        type=float,
        default=12.0,
        help="Cap used for accel and capped max-speed reporting; <=0 disables cap (default: 12.0)",
    )
    parser.add_argument(
        "--team-summary-all-tracks",
        action="store_true",
        help="If set, include unreliable tracks in team summary aggregates",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if args.min_reliable_frames <= 0:
        raise ValueError(
            f"--min-reliable-frames must be > 0, got {args.min_reliable_frames}"
        )
    if args.max_gap_frames <= 0:
        raise ValueError(f"--max-gap-frames must be > 0, got {args.max_gap_frames}")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError(f"--ema-alpha must be in (0, 1], got {args.ema_alpha}")
    if args.max_step_m_per_frame < 0:
        raise ValueError(
            f"--max-step-m-per-frame must be >= 0, got {args.max_step_m_per_frame}"
        )

    speed_cap_mps = args.speed_cap_mps if args.speed_cap_mps > 0 else None

    tracks_path = Path(args.tracks)
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_path}")

    output_dir = Path(args.output_dir) if args.output_dir else tracks_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    player_csv = output_dir / "player_summary.csv"
    team_csv = output_dir / "team_summary.csv"

    classes = _parse_classes(args.classes)
    df = pd.read_csv(tracks_path)
    _require_columns(
        df,
        ["frame", "track_id", "cls_name", "team_id", "pitch_x_m", "pitch_y_m"],
        str(tracks_path),
    )

    d = df[df["cls_name"].isin(classes)].copy()
    d["frame"] = pd.to_numeric(d["frame"], errors="coerce").fillna(-1).astype(int)
    d["track_id"] = pd.to_numeric(d["track_id"], errors="coerce").fillna(-1).astype(int)
    d["team_id"] = pd.to_numeric(d["team_id"], errors="coerce").fillna(-1).astype(int)
    d["pitch_x_m"] = pd.to_numeric(d["pitch_x_m"], errors="coerce")
    d["pitch_y_m"] = pd.to_numeric(d["pitch_y_m"], errors="coerce")
    d = d[(d["frame"] >= 0) & (d["track_id"] >= 0)].copy()
    d["_pitch_valid"] = _pitch_valid_mask(d)

    player_summary = build_player_summary(
        d,
        fps=args.fps,
        max_gap_frames=args.max_gap_frames,
        ema_alpha=args.ema_alpha,
        max_step_m_per_frame=args.max_step_m_per_frame,
        min_reliable_frames=args.min_reliable_frames,
        speed_cap_mps=speed_cap_mps,
    )
    player_summary.to_csv(
        player_csv,
        index=False,
        columns=[
            "track_id",
            "team_id",
            "frames_seen",
            "seconds_seen",
            "distance_m",
            "avg_speed_mps",
            "max_speed_mps",
            "max_speed_mps_capped",
            "avg_accel_mps2",
            "p95_accel_mps2",
            "top_speed_time_s",
            "is_reliable",
            "num_segments_used",
            "num_segments_skipped_gap",
            "num_segments_skipped_step",
            "num_speed_clipped",
        ],
    )

    team_summary = build_team_summary(
        filtered_df=d,
        player_summary=player_summary,
        reliable_only=(not args.team_summary_all_tracks),
    )
    team_summary.to_csv(
        team_csv,
        index=False,
        columns=[
            "team_id",
            "total_distance_m",
            "mean_player_distance_m",
            "median_player_distance_m",
            "mean_team_speed_mps",
            "avg_active_players_per_frame",
            "players_used",
        ],
    )

    print(f"Wrote player summary: {player_csv}")
    print(f"Wrote team summary: {team_csv}")
    print(f"Players: {len(player_summary)} | Teams: {len(team_summary)}")
    print(
        "Config: "
        f"min_reliable_frames={args.min_reliable_frames}, "
        f"max_gap_frames={args.max_gap_frames}, "
        f"ema_alpha={args.ema_alpha}, "
        f"max_step_m_per_frame={args.max_step_m_per_frame}, "
        f"speed_cap_mps={speed_cap_mps}"
    )


if __name__ == "__main__":
    main()
