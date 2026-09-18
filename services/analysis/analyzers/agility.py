from __future__ import annotations

from typing import Optional

import numpy as np

from services.analysis.analyzers.base import BaseTestAnalyzer


class AgilityAnalyzer(BaseTestAnalyzer):
    test_type = "AGILITY"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "AGILITY", "completion_time_s": None, "direction_changes": None, "path_efficiency": None, "acceleration_count": None, "deceleration_count": None, "confidence": 0.0, "calibration_status": "no_data"}

        has_calibration = calibration is not None and calibration.get("known_distances") is not None

        direction_changes = 0
        for i in range(2, len(tracks)):
            v1 = (tracks[i-1]["center"][0] - tracks[i-2]["center"][0], tracks[i-1]["center"][1] - tracks[i-2]["center"][1])
            v2 = (tracks[i]["center"][0] - tracks[i-1]["center"][0], tracks[i]["center"][1] - tracks[i-1]["center"][1])
            if np.sqrt(v1[0]**2 + v1[1]**2) > 0 and np.sqrt(v2[0]**2 + v2[1]**2) > 0:
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (np.sqrt(v1[0]**2+v1[1]**2) * np.sqrt(v2[0]**2+v2[1]**2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                if angle > 1.0:
                    direction_changes += 1

        result = {
            "test_type": "AGILITY",
            "completion_time_s": round(len(tracks) / fps, 3) if fps > 0 else None,
            "direction_changes": direction_changes,
            "path_efficiency": None,
            "acceleration_count": None,
            "deceleration_count": None,
            "confidence": 0.35,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }
        return result
