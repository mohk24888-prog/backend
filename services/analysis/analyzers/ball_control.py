from __future__ import annotations

from typing import Optional

from services.analysis.analyzers.base import BaseTestAnalyzer


class BallControlAnalyzer(BaseTestAnalyzer):
    test_type = "BALL_CONTROL"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "BALL_CONTROL", "touches": None, "control_duration_s": None, "failed_controls": None, "successful_controls": None, "confidence": 0.0, "calibration_status": "no_data"}

        has_calibration = calibration is not None

        touches = sum(1 for t in tracks if t.get("has_ball") is True)
        failed_controls = sum(1 for t in tracks if t.get("has_ball") is False and t.get("near_ball") is True)
        successful_controls = touches - failed_controls

        result = {
            "test_type": "BALL_CONTROL",
            "touches": touches if touches > 0 else None,
            "control_duration_s": round(touches / fps, 2) if touches > 0 and fps > 0 else None,
            "failed_controls": failed_controls if failed_controls > 0 else None,
            "successful_controls": successful_controls if successful_controls > 0 else None,
            "confidence": 0.25,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }
        return result
