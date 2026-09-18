from __future__ import annotations

from typing import Optional

import numpy as np

from services.analysis.analyzers.base import BaseTestAnalyzer


class DribblingAnalyzer(BaseTestAnalyzer):
    test_type = "DRIBBLING"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "DRIBBLING", "completion_time_s": None, "ball_losses": None, "path_efficiency": None, "distance_m": None, "confidence": 0.0, "calibration_status": "no_data"}

        has_calibration = calibration is not None and calibration.get("known_distances") is not None

        ball_losses = sum(1 for t in tracks if t.get("has_ball") is False)

        path_points = [(t["center"][0], t["center"][1]) for t in tracks]
        direct_dist = np.sqrt((path_points[-1][0] - path_points[0][0])**2 + (path_points[-1][1] - path_points[0][1])**2)
        total_dist = sum(np.sqrt((path_points[i][0]-path_points[i-1][0])**2 + (path_points[i][1]-path_points[i-1][1])**2) for i in range(1, len(path_points)))
        path_efficiency = round(direct_dist / total_dist, 3) if total_dist > 0 else None

        result = {
            "test_type": "DRIBBLING",
            "completion_time_s": round(len(tracks) / fps, 3) if fps > 0 else None,
            "ball_losses": ball_losses,
            "path_efficiency": path_efficiency,
            "distance_m": None,
            "confidence": 0.3,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }

        if has_calibration:
            meters_per_pixel = calibration.get("meters_per_pixel", 0)
            if meters_per_pixel and meters_per_pixel > 0:
                result["distance_m"] = round(total_dist * meters_per_pixel, 2)

        return result
