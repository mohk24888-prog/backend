from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.types import Track

_POSSESSION_CLASSES = {"Player", "Goalkeeper"}


@dataclass
class PossessionState:
    possessor_id: Optional[int] = None
    candidate_id: Optional[int] = None
    candidate_dist: Optional[float] = None


class PossessionTracker:
    """Track ball possession using foot-to-ball proximity with temporal smoothing."""

    def __init__(
        self,
        max_distance_px: float = 100.0,
        confirm_frames: int = 5,
        lose_frames: int = 5,
        switch_frames: int = 8,
    ):
        self.max_distance_px = max_distance_px
        self.confirm_frames = confirm_frames
        self.lose_frames = lose_frames
        self.switch_frames = switch_frames

        self.possessor_id: Optional[int] = None
        self._history: deque[Optional[int]] = deque(
            maxlen=max(confirm_frames, switch_frames, lose_frames)
        )
        self._no_ball_count: int = 0

    def update(
        self,
        tracks: list[Track],
        ball_xy: Optional[tuple[float, float]],
    ) -> Optional[int]:
        """Returns current possessor track_id or None."""
        if ball_xy is None:
            self._no_ball_count += 1
            self._history.append(None)
            if self._no_ball_count >= self.lose_frames:
                self.possessor_id = None
            return self.possessor_id

        self._no_ball_count = 0
        bx, by = ball_xy

        # Find nearest player/goalkeeper by foot position.
        best_id: Optional[int] = None
        best_dist = float("inf")
        for track in tracks:
            if track.cls_name not in _POSSESSION_CLASSES:
                continue
            if track.track_id < 0:
                continue
            fx, fy = track.foot_position
            dist = np.hypot(fx - bx, fy - by)
            if dist < best_dist:
                best_dist = dist
                best_id = track.track_id

        candidate = best_id if best_dist <= self.max_distance_px else None
        self._history.append(candidate)

        if self.possessor_id is None:
            # Gain possession: candidate must appear in last confirm_frames entries.
            if candidate is not None and len(self._history) >= self.confirm_frames:
                recent = list(self._history)[-self.confirm_frames :]
                if all(c == candidate for c in recent):
                    self.possessor_id = candidate
        else:
            # Check if current possessor is still the candidate.
            if candidate == self.possessor_id:
                return self.possessor_id

            # Check for loss: consecutive Nones in recent history.
            recent_all = list(self._history)[-self.lose_frames :]
            if len(recent_all) >= self.lose_frames and all(
                c is None for c in recent_all
            ):
                self.possessor_id = None
                return self.possessor_id

            # Check for switch: another player dominates last switch_frames.
            if candidate is not None and len(self._history) >= self.switch_frames:
                recent_switch = list(self._history)[-self.switch_frames :]
                if all(c == candidate for c in recent_switch):
                    self.possessor_id = candidate

        return self.possessor_id


class TeamPossessionTracker:
    """Map player-level possession to team-level with smoothing and frame counting."""

    def __init__(self, hold_frames: int = 10):
        self.hold_frames = hold_frames
        self.team_id: Optional[int] = None
        self._no_team_count: int = 0
        self.team_frame_counts: dict[int, int] = {}

    def update(self, possessor_id: Optional[int], tracks: list[Track]) -> Optional[int]:
        """Returns current possessing team_id or None. Increments frame counters."""
        frame_team: Optional[int] = None
        if possessor_id is not None:
            for track in tracks:
                if track.track_id == possessor_id and track.team_id is not None:
                    frame_team = track.team_id
                    break

        if frame_team is not None:
            self.team_id = frame_team
            self._no_team_count = 0
        else:
            self._no_team_count += 1
            if self._no_team_count >= self.hold_frames:
                self.team_id = None

        if self.team_id is not None:
            self.team_frame_counts[self.team_id] = (
                self.team_frame_counts.get(self.team_id, 0) + 1
            )

        return self.team_id

    def percentages(self) -> dict[int, float]:
        """Return {team_id: percentage} for all teams seen."""
        total = sum(self.team_frame_counts.values())
        if total == 0:
            return {}
        return {
            tid: count / total * 100.0 for tid, count in self.team_frame_counts.items()
        }
