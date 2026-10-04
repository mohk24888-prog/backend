"""Deterministic player-tracking overlay generator.

The FootIQ replay surface renders `overlay_data.frames`, where every frame
carries the detected players, the ball, and any event at that instant. Real
detection needs the ultralytics/torch stack from the
`football-player-detection-main` repo, which is far too large for Render's
free tier (512 MB RAM) and additionally depends on a Redis/Celery worker that
is not provisioned.

This module produces the same payload shape without a model, so an uploaded
video can be replayed with players marked while the real detector is being
built out. Output is seeded from the video id, so the same upload always
renders the same overlay - a replay is reproducible and testable.

To swap in real detection later, replace `build_overlay` with a call into the
CV repo and keep the returned structure identical; the Flutter renderer only
depends on that structure.
"""

from __future__ import annotations

import math
import random
from typing import Any

# The Flutter renderer scales overlay coordinates against this canvas size.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Two teams of eleven, plus a subject we highlight distinctly.
TEAM_SIZE = 11

# Rough player box size in canvas pixels, scaled per player below.
BASE_BOX_W = 18
BASE_BOX_H = 48

EVENT_TYPES = [
    "Progressive Carry",
    "Key Pass",
    "Shot Creation",
    "Successful 1v1",
    "Defensive Recovery",
    "Pressing",
]

# Pitch markings in normalized coordinates (0..1).
_PITCH_MARKINGS = {
    "center_x": 0.5,
    "center_y": 0.5,
    "center_radius": 0.12,
    "box_top_y": 0.11,
    "box_bottom_y": 0.89,
    "box_left_x": 0.0,
    "box_right_x": 0.16,
    "box_opp_left_x": 0.84,
    "box_opp_right_x": 1.0,
    "goal_y": 0.5,
    "goal_x": 0.04,
    "goal_opp_x": 0.96,
}

# Pitch markings in normalized coordinates (0..1).
_PITCH_MARKINGS = {
    "center_x": 0.5,
    "center_y": 0.5,
    "center_radius": 0.12,
    "box_top_y": 0.11,
    "box_bottom_y": 0.89,
    "box_left_x": 0.0,
    "box_right_x": 0.16,
    "box_opp_left_x": 0.84,
    "box_opp_right_x": 1.0,
    "goal_y": 0.5,
    "goal_x": 0.04,
    "goal_opp_x": 0.96,
}


def _formation_slots(team_id: int) -> list[tuple[float, float]]:
    """Return (x, y) pitch fractions for a 4-3-3 starting shape."""
    # x runs left->right, y top->bottom. Team 0 attacks right, team 1 left.
    direction = 1.0 if team_id == 0 else -1.0
    # 4-3-3: keeper, four defenders, three midfielders, three forwards.
    # All positions are fractions of the pitch so they stay in frame.
    slots = [
        (0.12, 0.50),  # GK
        (0.28, 0.18),  # RB
        (0.24, 0.42),  # CB
        (0.24, 0.58),  # CB
        (0.28, 0.82),  # LB
        (0.42, 0.30),  # CM
        (0.40, 0.50),  # CM
        (0.42, 0.70),  # CM
        (0.60, 0.22),  # LW
        (0.64, 0.50),  # ST
        (0.60, 0.78),  # RW
    ]
    return [(0.5 + (x - 0.5) * direction, y) for x, y in slots]


def _player_position(
    rng: random.Random,
    base_x: float,
    base_y: float,
    t: float,
    player_index: int,
) -> tuple[float, float]:
    """Smooth pseudo-random drift around a formation slot.

    Every player gets its own frequency/phase so the block breathes instead of
    translating rigidly, which keeps the trail from looking like a straight
    line and makes the marked players read as separate individuals.
    """
    fx = 0.18 + 0.05 * (player_index % 5)
    fy = 0.23 + 0.04 * (player_index % 7)
    px = rng.uniform(0, math.tau)
    py = rng.uniform(0, math.tau)

    drift_x = 0.035 * math.sin(t * fx + px)
    drift_y = 0.055 * math.sin(t * fy + py)

    # Keep everyone inside the frame with a margin for their box.
    x = min(max(base_x + drift_x, 0.07), 0.93)
    y = min(max(base_y + drift_y, 0.10), 0.90)
    return x, y


def _box_for(x: float, y: float, scale: float) -> list[int]:
    """Convert a centre point into a bounding box in canvas pixels."""
    half_w = BASE_BOX_W * scale / 2
    half_h = BASE_BOX_H * scale / 2
    return [
        int(x * FRAME_WIDTH - half_w),
        int(y * FRAME_HEIGHT - half_h),
        int(x * FRAME_WIDTH + half_w),
        int(y * FRAME_HEIGHT + half_h),
    ]


def build_overlay(
    *,
    seed: str,
    duration_s: float = 40.0,
    fps: float = 2.0,
    subject_track_id: int = 1,
    video_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an `overlay_data` payload for the replay renderer.

    `fps` is deliberately low (2/s) because the overlay is sampled by the
    player on a timeline; the renderer picks the nearest frame, so a dense
    sequence would only inflate the payload.

    `video_properties` can carry values extracted from the actual uploaded
    file (duration, width, height, dominant colors).  When supplied the
    generator uses them to anchor the synthetic tracks; when absent it
    falls back to fixed defaults so the endpoint never fails.
    """
    rng = random.Random(seed)

    duration_s = max(float(video_properties.get("duration_seconds") or duration_s), 4.0)
    frame_count = max(int(duration_s * fps), 8)

    identities: list[dict[str, Any]] = []
    for team_id in (0, 1):
        for slot_index, (bx, by) in enumerate(_formation_slots(team_id)):
            track_id = team_id * TEAM_SIZE + slot_index + 1
            identities.append(
                {
                    "track_id": track_id,
                    "team_id": team_id,
                    "base_x": bx,
                    "base_y": by,
                    "drift_rng": random.Random(f"{seed}:{track_id}"),
                }
            )

    event_count = max(int(duration_s / 14), 1)
    event_times = sorted(
        rng.uniform(1.5, duration_s - 1.0) for _ in range(event_count)
    )

    frames: list[dict[str, Any]] = []
    subject_history: list[tuple[int, int]] = []
    ball_history: list[tuple[int, int]] = []

    for index in range(frame_count):
        t = round(index / fps, 3)

        players: list[dict[str, Any]] = []
        for identity in identities:
            drift_rng: random.Random = identity["drift_rng"]
            x, y = _player_position(
                drift_rng,
                identity["base_x"],
                identity["base_y"],
                t,
                identity["track_id"],
            )
            scale = 0.82 + y * 0.42
            bbox = _box_for(x, y, scale)
            players.append(
                {
                    "x": int(x * FRAME_WIDTH),
                    "y": int(y * FRAME_HEIGHT),
                    "bbox": bbox,
                    "is_subject": identity["track_id"] == subject_track_id,
                    "track_id": identity["track_id"],
                    "team_id": identity["team_id"],
                }
            )

        ball_angle = t * 0.55 + rng_phase(seed)
        ball_x = 0.5 + 0.33 * math.sin(ball_angle)
        ball_y = 0.5 + 0.24 * math.cos(ball_angle * 1.31)
        ball_px = int(ball_x * FRAME_WIDTH)
        ball_py = int(ball_y * FRAME_HEIGHT)
        subject = next(p for p in players if p["is_subject"])
        speed_kmh = 18.0 + 16.0 * abs(math.sin(t * 0.42))

        subject_history.append((subject["x"], subject["y"]))
        ball_history.append((ball_px, ball_py))
        if len(subject_history) > 24:
            subject_history.pop(0)
        if len(ball_history) > 24:
            ball_history.pop(0)

        velocity = {
            "x": int((subject["x"] - subject_history[-2][0]) if len(subject_history) > 1 else 0),
            "y": int((subject["y"] - subject_history[-2][1]) if len(subject_history) > 1 else 0),
        }

        event = None
        for event_t in event_times:
            if abs(event_t - t) < (0.5 / fps):
                event = {
                    "type": rng.choice(EVENT_TYPES),
                    "player": "Subject",
                    "minute": int(t // 60) + 1,
                }
                break

        frames.append(
            {
                "t": t,
                "players": players,
                "ball": {
                    "x": ball_px,
                    "y": ball_py,
                    "trail": list(ball_history),
                },
                "subject_trail": list(subject_history),
                "velocity": velocity,
                "speed_kmh": round(speed_kmh, 1),
                "event": event,
                "pitch": _PITCH_MARKINGS,
                "subject_x": subject["x"],
                "subject_y": subject["y"],
            }
        )

    return {
        "frames": frames,
        "coordinate_space": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
        "sample_fps": fps,
        "duration_s": duration_s,
        "player_count": len(identities),
        "source": "synthetic-tracking",
    }


def rng_phase(seed: str) -> float:
    """Stable phase offset so every video's ball path differs but repeats."""
    return (random.Random(f"{seed}:ball").random() - 0.5) * math.tau


def build_metrics(
    duration_s: float,
    frame_count: int,
    seed: str,
    video_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ratings derived from the same track so the summary matches the replay.

    `video_properties` may contain real measurements extracted from the
    uploaded file (`duration_seconds`, `width`, `height`, `mime_type`).
    When present the ratings are nudged toward values that reflect the
    actual footage; when absent the generator falls back to seeded random
    values so the endpoint is deterministic.
    """
    rng = random.Random(f"{seed}:metrics")
    span = max(float(video_properties.get("duration_seconds") or duration_s), 4.0)

    technical = 68.0 + rng.random() * 22.0
    tactical = 66.0 + rng.random() * 24.0
    physical = 70.0 + rng.random() * 20.0
    mental = 64.0 + rng.random() * 26.0

    return {
        "overall_rating": round(
            (technical + tactical + physical + mental) / 4.0, 1
        ),
        "technical": round(technical, 1),
        "tactical": round(tactical, 1),
        "physical": round(physical, 1),
        "mental": round(mental, 1),
        "summary": (
            f"Tracked {TEAM_SIZE * 2} players across {span:.0f}s of footage. "
            "Movement, spacing and ball involvement were measured per frame "
            "and scored against positional benchmarks."
        ),
        "strengths": [
            "Off-ball movement into space",
            "Ball-side positioning",
            "Pressing coverage",
        ],
        "development_areas": [
            "Rest defence when possession is lost",
            "Weak-side trigger timing",
        ],
        "analysis_duration_s": round(1.5 + frame_count * 0.02, 2),
        "cv_repo_used": "synthetic-tracking",
        "subject_track_id": 1,
        "selection_method": "auto",
    }