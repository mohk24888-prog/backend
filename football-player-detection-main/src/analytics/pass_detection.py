from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.core.types import Track


@dataclass(frozen=True)
class PassEvent:
    frame_idx: int
    passer_id: int
    receiver_id: int
    team_id: int
    passer_pos_cm: tuple[float, float]
    receiver_pos_cm: tuple[float, float]
    distance_cm: float


@dataclass(frozen=True)
class TurnoverEvent:
    frame_idx: int
    losing_player_id: int
    gaining_player_id: int
    losing_team_id: int
    gaining_team_id: int
    losing_pos_cm: tuple[float, float]
    gaining_pos_cm: tuple[float, float]
    distance_cm: float


class PassDetector:
    """Detect completed passes by tracking same-team possessor transitions."""

    def __init__(
        self,
        min_pass_distance_cm: float = 300.0,
        max_pass_distance_cm: float = 6000.0,
    ):
        self.min_pass_distance_cm = min_pass_distance_cm
        self.max_pass_distance_cm = max_pass_distance_cm

        self._prev_possessor_id: Optional[int] = None
        self._prev_team_id: Optional[int] = None
        self._prev_pos_cm: Optional[tuple[float, float]] = None
        self.passes: list[PassEvent] = []
        self.turnovers: list[TurnoverEvent] = []

    def update(
        self,
        possessor_id: Optional[int],
        tracks: list[Track],
        frame_idx: int,
    ) -> Optional[PassEvent]:
        """Update with current frame state. Returns a PassEvent if a pass was detected."""
        if possessor_id is None:
            return None

        # Look up current possessor info from tracks.
        cur_team_id: Optional[int] = None
        cur_pos_cm: Optional[tuple[float, float]] = None
        for track in tracks:
            if track.track_id == possessor_id:
                cur_team_id = track.team_id
                if track.pitch_x_cm is not None and track.pitch_y_cm is not None:
                    cur_pos_cm = (track.pitch_x_cm, track.pitch_y_cm)
                break

        pass_event: Optional[PassEvent] = None

        # Check for possessor change.
        if (
            self._prev_possessor_id is not None
            and possessor_id != self._prev_possessor_id
            and cur_team_id is not None
            and self._prev_team_id is not None
            and cur_team_id == self._prev_team_id
            and cur_pos_cm is not None
            and self._prev_pos_cm is not None
        ):
            dx = cur_pos_cm[0] - self._prev_pos_cm[0]
            dy = cur_pos_cm[1] - self._prev_pos_cm[1]
            dist = math.hypot(dx, dy)
            if self.min_pass_distance_cm <= dist <= self.max_pass_distance_cm:
                pass_event = PassEvent(
                    frame_idx=frame_idx,
                    passer_id=self._prev_possessor_id,
                    receiver_id=possessor_id,
                    team_id=cur_team_id,
                    passer_pos_cm=self._prev_pos_cm,
                    receiver_pos_cm=cur_pos_cm,
                    distance_cm=dist,
                )
                self.passes.append(pass_event)
        elif (
            self._prev_possessor_id is not None
            and possessor_id != self._prev_possessor_id
            and cur_team_id is not None
            and self._prev_team_id is not None
            and cur_team_id != self._prev_team_id
            and cur_pos_cm is not None
            and self._prev_pos_cm is not None
        ):
            dx = cur_pos_cm[0] - self._prev_pos_cm[0]
            dy = cur_pos_cm[1] - self._prev_pos_cm[1]
            dist = math.hypot(dx, dy)
            if self.min_pass_distance_cm <= dist <= self.max_pass_distance_cm:
                turnover = TurnoverEvent(
                    frame_idx=frame_idx,
                    losing_player_id=self._prev_possessor_id,
                    gaining_player_id=possessor_id,
                    losing_team_id=self._prev_team_id,
                    gaining_team_id=cur_team_id,
                    losing_pos_cm=self._prev_pos_cm,
                    gaining_pos_cm=cur_pos_cm,
                    distance_cm=dist,
                )
                self.turnovers.append(turnover)

        # Update state: track current possessor's latest position.
        self._prev_possessor_id = possessor_id
        self._prev_team_id = cur_team_id
        if cur_pos_cm is not None:
            self._prev_pos_cm = cur_pos_cm

        return pass_event

    def summary(self) -> dict[int, dict[str, int]]:
        """Return per-team pass counts."""
        result: dict[int, dict[str, int]] = {}
        for p in self.passes:
            if p.team_id not in result:
                result[p.team_id] = {"completed": 0}
            result[p.team_id]["completed"] += 1
        return result

    def turnover_summary(self) -> dict[int, dict[str, int]]:
        """Return per-team turnover counts keyed by losing team."""
        result: dict[int, dict[str, int]] = {}
        for t in self.turnovers:
            if t.losing_team_id not in result:
                result[t.losing_team_id] = {"lost": 0}
            result[t.losing_team_id]["lost"] += 1
        return result
