from __future__ import annotations

from typing import Optional

from services.analysis.analyzers.base import BaseTestAnalyzer


class PassingAnalyzer(BaseTestAnalyzer):
    test_type = "PASSING"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "PASSING", "attempts": None, "successful_passes": None, "failed_passes": None, "accuracy_pct": None, "confidence": 0.0, "calibration_status": "no_data"}

        has_calibration = calibration is not None and calibration.get("target_positions") is not None

        attempts = len(tracks)
        successful_passes = sum(1 for t in tracks if t.get("near_target") is True)
        failed_passes = attempts - successful_passes
        accuracy = round(successful_passes / attempts * 100, 2) if attempts > 0 else None

        result = {
            "test_type": "PASSING",
            "attempts": attempts if attempts > 0 else None,
            "successful_passes": successful_passes if successful_passes > 0 else None,
            "failed_passes": failed_passes if failed_passes > 0 else None,
            "accuracy_pct": accuracy,
            "confidence": 0.28,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }
        return result
