# src/io/csv_writer.py
import csv
from pathlib import Path
from typing import Optional

from src.core.types import Track

HEADER = [
    "frame",
    "track_id",
    "cls_id",
    "cls_name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "fx",
    "fy",
    "width",
    "height",
    "team_id",
    "team_conf",
    "pitch_x_m",
    "pitch_y_m",
    "pitch_point_valid",
    "homography_valid",
    "homography_used_fallback",
    "homography_stale_frames",
    "homography_status",
    "homography_reject_reason",
    "homography_inlier_ratio",
    "homography_reproj_error_m",
]


class CsvWriter:
    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)

    def write_frame(
        self,
        frame_idx: int,
        tracks: list[Track],
        width: int,
        height: int,
        homography_valid: bool = False,
        homography_used_fallback: bool = False,
        homography_stale_frames: int = 0,
        homography_status: str = "invalid",
        homography_reject_reason: str = "not_estimated",
        homography_inlier_ratio: Optional[float] = None,
        homography_reproj_error_m: Optional[float] = None,
    ) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            cx, cy = t.center
            fx, fy = t.foot_position
            has_pitch_point = t.pitch_x_m is not None and t.pitch_y_m is not None

            self._writer.writerow(
                [
                    frame_idx,
                    t.track_id,
                    t.cls_id,
                    t.cls_name,
                    f"{t.conf:.6f}",
                    x1,
                    y1,
                    x2,
                    y2,
                    f"{cx:.2f}",
                    f"{cy:.2f}",
                    f"{fx:.2f}",
                    f"{fy:.2f}",
                    width,
                    height,
                    t.team_id if t.team_id is not None else -1,
                    f"{t.team_conf:.2f}" if t.team_conf is not None else -1,
                    f"{t.pitch_x_m:.3f}" if t.pitch_x_m is not None else -1,
                    f"{t.pitch_y_m:.3f}" if t.pitch_y_m is not None else -1,
                    int(has_pitch_point),
                    int(homography_valid),
                    int(homography_used_fallback),
                    homography_stale_frames,
                    homography_status,
                    homography_reject_reason,
                    f"{homography_inlier_ratio:.4f}"
                    if homography_inlier_ratio is not None
                    and homography_inlier_ratio >= 0
                    else -1,
                    f"{homography_reproj_error_m:.4f}"
                    if homography_reproj_error_m is not None
                    and homography_reproj_error_m >= 0
                    else -1,
                ]
            )

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
