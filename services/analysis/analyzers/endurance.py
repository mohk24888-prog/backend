from __future__ import annotations

from typing import Optional

import numpy as np

from services.analysis.analyzers.base import BaseTestAnalyzer


class EnduranceAnalyzer(BaseTestAnalyzer):
    test_type = "ENDURANCE"

    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        if not tracks:
            return {"test_type": "ENDURANCE", "distance_m": None, "elapsed_time_s": None, "avg_speed_ms": None, "max_speed_ms": None, "repetition_count": None, "confidence": 0.0, "calibration_status": "no_data"}

        has_calibration = calibration is not None and calibration.get("known_distances") is not None

        distances = []
        speeds = []
        for i in range(1, len(tracks)):
            px_dist = np.sqrt((tracks[i]["center"][0] - tracks[i-1]["center"][0])**2 + (tracks[i]["center"][1] - tracks[i-1]["center"][1])**2)
            frame_diff = tracks[i].get("frame_id", 0) - tracks[i-1].get("frame_id", 0)
            if frame_diff > 0 and fps > 0:
                dt = frame_diff / fps
                if has_calibration and calibration.get("meters_per_pixel"):
                    distances.append(px_dist * calibration["meters_per_pixel"])
                    speeds.append(distances[-1] / dt)
                else:
                    distances.append(None)
                    speeds.append(None)
            else:
                distances.append(None)
                speeds.append(None)

        real_distances = [d for d in distances if d is not None]
        real_speeds = [s for s in speeds if s is not None]

        speed_zones = {"walking": 0, "jogging": 0, "running": 0, "sprinting": 0}
        for s in real_speeds:
            if s < 1.5:
                speed_zones["walking"] += 1
            elif s < 3.5:
                speed_zones["jogging"] += 1
            elif s < 6.0:
                speed_zones["running"] += 1
            else:
                speed_zones["sprinting"] += 1

        result = {
            "test_type": "ENDURANCE",
            "distance_m": round(sum(real_distances), 2) if real_distances else None,
            "elapsed_time_s": round(len(tracks) / fps, 3) if fps > 0 else None,
            "avg_speed_ms": round(sum(real_speeds) / len(real_speeds), 2) if real_speeds else None,
            "max_speed_ms": round(max(real_speeds), 2) if real_speeds else None,
            "repetition_count": None,
            "speed_zones": speed_zones,
            "confidence": 0.35,
            "calibration_status": "calibrated" if has_calibration else "insufficient_calibration",
        }
        return result
