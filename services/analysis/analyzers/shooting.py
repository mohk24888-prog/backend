from __future__ import annotations

from typing import Optional

from services.analysis.analyzers.base import BaseTestAnalyzer


class ShootingAnalyzer(BaseTestAnalyzer):
    test_type = "SHOOTING"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "SHOOTING", "shots_attempted": None, "shots_on_target": None, "goals": None, "target_accuracy_pct": None, "ball_speed_ms": None, "confidence": 0.0, "calibration_status": "no_data"}

        has_calibration = calibration is not None and calibration.get("target_positions") is not None

        shots_attempted = len(tracks)
        shots_on_target = sum(1 for t in tracks if t.get("near_target") is True and t.get("has_ball") is True)
        goals = sum(1 for t in tracks if t.get("in_goal_zone") is True and t.get("has_ball") is True)
        target_accuracy = round(shots_on_target / shots_attempted * 100, 2) if shots_attempted > 0 else None

        result = {
            "test_type": "SHOOTING",
            "shots_attempted": shots_attempted if shots_attempted > 0 else None,
            "shots_on_target": shots_on_target if shots_on_target > 0 else None,
            "goals": goals if goals > 0 else None,
            "target_accuracy_pct": target_accuracy,
            "ball_speed_ms": None,
            "confidence": 0.2,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }
        return result
