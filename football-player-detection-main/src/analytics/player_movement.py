import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from src.core.types import Track


@dataclass
class _TrackMovementState:
    track_id: int
    cls_name: str
    team_id: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    tracked_frames: int = 0
    total_distance_m: float = 0.0
    total_elapsed_s: float = 0.0
    peak_speed_mps: float = 0.0
    current_speed_mps: Optional[float] = None
    last_frame: Optional[int] = None
    last_point_m: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PlayerMovementSummary:
    track_id: int
    cls_name: str
    team_id: int
    start_frame: int
    end_frame: int
    tracked_frames: int
    minutes_tracked: float
    frame_coverage: float
    frame_coverage_pct: float
    total_distance_m: float
    avg_speed_mps: float
    peak_speed_mps: float


class PlayerMovementAnalyzer:
    """
    Aggregates per-track movement metrics from smoothed pitch coordinates.
    """

    HEADER = [
        "track_id",
        "cls_name",
        "team_id",
        "start_frame",
        "end_frame",
        "tracked_frames",
        "minutes_tracked",
        "frame_coverage",
        "frame_coverage_pct",
        "total_distance_m",
        "avg_speed_mps",
        "peak_speed_mps",
    ]

    def __init__(
        self,
        fps: float,
        tracked_classes: Iterable[str] = ("Player", "Goalkeeper"),
    ):
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = float(fps)
        self.tracked_classes = set(tracked_classes)
        self._states: Dict[int, _TrackMovementState] = {}
        self._total_frames_processed = 0

    def update(self, frame_idx: int, tracks: list[Track]) -> None:
        self._total_frames_processed = max(self._total_frames_processed, frame_idx + 1)

        for track in tracks:
            if track.track_id < 0:
                continue
            if track.cls_name not in self.tracked_classes:
                continue
            if track.pitch_x_m is None or track.pitch_y_m is None:
                continue

            point_m = np.array([track.pitch_x_m, track.pitch_y_m], dtype=np.float32)
            state = self._states.get(track.track_id)
            if state is None:
                state = _TrackMovementState(
                    track_id=track.track_id, cls_name=track.cls_name
                )
                self._states[track.track_id] = state

            state.cls_name = track.cls_name
            if track.team_id is not None:
                state.team_id = int(track.team_id)

            if state.start_frame is None:
                state.start_frame = frame_idx
            state.end_frame = frame_idx
            state.tracked_frames += 1
            state.current_speed_mps = None

            if (
                state.last_frame is not None
                and state.last_point_m is not None
                and frame_idx > state.last_frame
            ):
                distance_m = float(np.linalg.norm(point_m - state.last_point_m))
                dt_s = (frame_idx - state.last_frame) / self.fps
                if np.isfinite(distance_m) and np.isfinite(dt_s) and dt_s > 0:
                    state.total_distance_m += distance_m
                    state.total_elapsed_s += dt_s
                    speed_mps = distance_m / dt_s
                    state.current_speed_mps = speed_mps
                    if speed_mps > state.peak_speed_mps:
                        state.peak_speed_mps = speed_mps

            state.last_point_m = point_m
            state.last_frame = frame_idx

    def _to_summary(
        self, state: _TrackMovementState, total_frames: int
    ) -> PlayerMovementSummary:
        tracked_time_s = state.tracked_frames / self.fps
        minutes_tracked = tracked_time_s / 60.0
        avg_speed_mps = (
            state.total_distance_m / state.total_elapsed_s
            if state.total_elapsed_s > 0
            else 0.0
        )
        frame_coverage = (
            (state.tracked_frames / total_frames) if total_frames > 0 else 0.0
        )

        return PlayerMovementSummary(
            track_id=state.track_id,
            cls_name=state.cls_name,
            team_id=state.team_id if state.team_id is not None else -1,
            start_frame=state.start_frame if state.start_frame is not None else -1,
            end_frame=state.end_frame if state.end_frame is not None else -1,
            tracked_frames=state.tracked_frames,
            minutes_tracked=minutes_tracked,
            frame_coverage=frame_coverage,
            frame_coverage_pct=frame_coverage * 100.0,
            total_distance_m=state.total_distance_m,
            avg_speed_mps=avg_speed_mps,
            peak_speed_mps=state.peak_speed_mps,
        )

    def get_summaries(
        self, total_frames: Optional[int] = None
    ) -> list[PlayerMovementSummary]:
        if total_frames is None:
            total_frames = self._total_frames_processed
        total_frames = max(0, int(total_frames))

        summaries = [
            self._to_summary(state, total_frames) for state in self._states.values()
        ]
        summaries.sort(key=lambda s: s.track_id)
        return summaries

    def get_live_speed_mps(
        self, track_id: int, current_frame_idx: Optional[int] = None
    ) -> Optional[float]:
        state = self._states.get(track_id)
        if state is None:
            return None
        if current_frame_idx is not None and state.last_frame != current_frame_idx:
            return None
        if state.current_speed_mps is None:
            return None
        return float(state.current_speed_mps)

    def write_summary(
        self, output_csv_path: str, total_frames: Optional[int] = None
    ) -> None:
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summaries = self.get_summaries(total_frames=total_frames)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
            for s in summaries:
                writer.writerow(
                    [
                        s.track_id,
                        s.cls_name,
                        s.team_id,
                        s.start_frame,
                        s.end_frame,
                        s.tracked_frames,
                        f"{s.minutes_tracked:.5f}",
                        f"{s.frame_coverage:.6f}",
                        f"{s.frame_coverage_pct:.3f}",
                        f"{s.total_distance_m:.3f}",
                        f"{s.avg_speed_mps:.3f}",
                        f"{s.peak_speed_mps:.3f}",
                    ]
                )
