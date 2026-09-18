from __future__ import annotations

from typing import Optional

import numpy as np

from services.analysis.analyzers.base import BaseTestAnalyzer


class SprintAnalyzer(BaseTestAnalyzer):
    test_type = "SPRINT"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "SPRINT", "test_distance_m": None, "completion_time_s": None, "average_speed_ms": None, "max_speed_ms": None, "acceleration_ms2": None, "deceleration_ms2": None, "confidence": 0.0, "calibration_status": "no_data"}

        calibration = calibration or {}
        meters_per_pixel = calibration.get("meters_per_pixel")
        has_calibration = meters_per_pixel is not None and meters_per_pixel > 0

        distances = []
        speeds = []
        accelerations = []
        for i in range(1, len(tracks)):
            prev = tracks[i - 1]
            curr = tracks[i]
            px_dist = np.sqrt((curr["center"][0] - prev["center"][0]) ** 2 + (curr["center"][1] - prev["center"][1]) ** 2)
            if has_calibration:
                distances.append(px_dist * meters_per_pixel)
            else:
                distances.append(None)

            frame_diff = curr.get("frame_id", 0) - prev.get("frame_id", 0)
            if frame_diff > 0 and fps > 0:
                dt = frame_diff / fps
                if distances[-1] is not None and dt > 0:
                    speeds.append(distances[-1] / dt)
                else:
                    speeds.append(None)

        real_distances = [d for d in distances if d is not None]
        real_speeds = [s for s in speeds if s is not None]

        result = {
            "test_type": "SPRINT",
            "test_distance_m": round(sum(real_distances), 2) if real_distances else None,
            "completion_time_s": round(len(tracks) / fps, 3) if tracks and fps > 0 else None,
            "average_speed_ms": round(sum(real_speeds) / len(real_speeds), 2) if real_speeds else None,
            "max_speed_ms": round(max(real_speeds), 2) if real_speeds else None,
            "acceleration_ms2": None,
            "deceleration_ms2": None,
            "confidence": 0.4,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }

        if real_speeds and len(real_speeds) >= 2:
            result["acceleration_ms2"] = round((real_speeds[1] - real_speeds[0]) * fps, 2)
            result["deceleration_ms2"] = round((real_speeds[-1] - real_speeds[-2]) * fps, 2)

        if not has_calibration:
            result["test_distance_m"] = None
            result["average_speed_ms"] = None
            result["max_speed_ms"] = None

        return result
