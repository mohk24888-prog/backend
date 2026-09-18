from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from core.config import settings


class DeviceManager:
    @staticmethod
    def get_device() -> torch.device:
        if settings.device == "cpu":
            return torch.device("cpu")
        if settings.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def get_device_name() -> str:
        device = DeviceManager.get_device()
        if device.type == "cuda":
            return f"CUDA:{device.index}"
        return "CPU"


class VideoPreprocessor:
    def __init__(self, raw_dir: Path = settings.raw_dir):
        self.raw_dir = raw_dir

    def get_video_path(self, video_id: str) -> Optional[Path]:
        candidate = self.raw_dir / video_id
        if candidate.exists():
            return candidate
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            c = self.raw_dir / f"{video_id}{ext}"
            if c.exists():
                return c
        return None

    def extract_frames(
        self,
        video_path: Path,
        frame_skip: int = settings.frame_skip,
        max_frames: Optional[int] = None,
    ) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames: list[np.ndarray] = []
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_skip == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            frame_id += 1
        cap.release()
        logger.info("Extracted {} frames", len(frames))
        return frames


class PlayerDetector:
    def __init__(self, model_name: str = settings.yolo_model):
        self.model = YOLO(model_name)
        self.device = DeviceManager.get_device()
        logger.info("Loaded YOLO model: {} on {}", model_name, self.device)

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame,
            conf=settings.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "center": [
                            int((x1 + x2) / 2),
                            int((y1 + y2) / 2),
                        ],
                    }
                )
        return detections


class Tracker:
    def __init__(self):
        self.tracks: dict[int, list[dict]] = {}
        self.next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        tracked = []
        for det in detections:
            best_id = None
            best_iou = 0.5
            for tid, path in self.tracks.items():
                last = path[-1]
                iou = self._iou(det["bbox"], last["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is not None:
                self.tracks[best_id].append(det)
                tracked.append({"track_id": best_id, **det})
            else:
                self.tracks[self.next_id] = [det]
                tracked.append({"track_id": self.next_id, **det})
                self.next_id += 1
        return tracked

    @staticmethod
    def _iou(a: list, b: list) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


class MetricsCalculator:
    @staticmethod
    def physical_metrics(
        tracks: list[dict], fps: float = 25.0
    ) -> dict:
        if not tracks:
            return {"error": "no_tracks"}
        distances = []
        for track in tracks:
            path = track.get("path", [])
            if len(path) < 2:
                continue
            for i in range(1, len(path)):
                dx = path[i][0] - path[i - 1][0]
                dy = path[i][1] - path[i - 1][1]
                distances.append(np.sqrt(dx**2 + dy**2))
        if not distances:
            return {"error": "insufficient_data"}
        total_distance = float(sum(distances) * 0.04)
        avg_speed = total_distance / (len(tracks) / fps) if fps else 0
        return {
            "total_distance_m": round(total_distance, 2),
            "avg_speed_ms": round(avg_speed, 2),
            "confidence": "low" if len(tracks) < 5 else "medium",
        }

    @staticmethod
    def tactical_metrics(tracks: list[dict]) -> dict:
        pressing_count = sum(1 for t in tracks if t.get("is_pressing", False))
        return {
            "press_count": pressing_count,
            "ppda_contribution": round(pressing_count * 0.8, 2),
            "confidence": "low" if len(tracks) < 5 else "medium",
        }

    @staticmethod
    def heatmap(tracks: list[dict], grid_size: tuple[int, int] = (12, 8)) -> list[dict]:
        grid = np.zeros(grid_size)
        for track in tracks:
            cx, cy = track.get("center", [0, 0])
            gx = int(cx / 640 * grid_size[0])
            gy = int(cy / 480 * grid_size[1])
            if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
                grid[gy, gx] += 1
        points = []
        for y in range(grid_size[1]):
            for x in range(grid_size[0]):
                if grid[y, x] > 0:
                    points.append(
                        {"x": x / (grid_size[0] - 1), "y": y / (grid_size[1] - 1), "intensity": float(grid[y, x])}
                    )
        return points
