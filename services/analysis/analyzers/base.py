from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseTestAnalyzer(ABC):
    test_type: str = ""

    @abstractmethod
    def analyze(
        self,
        tracks: list[dict],
        calibration: Optional[dict] = None,
        video_path: Optional[str] = None,
        fps: float = 25.0,
    ) -> dict:
        """Analyze tracked player data for a specific test type.

        Returns a dict of metrics. Values may be None if measurement is unavailable.
        """
        pass

    @staticmethod
    def _tracking_confidence(tracks: list[dict]) -> float:
        if not tracks:
            return 0.0
        tracked_frames = sum(1 for t in tracks if t.get("track_id") is not None)
        return round(tracked_frames / len(tracks), 3)