from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PitchKeypointResult:
    xy: np.ndarray
    conf: np.ndarray
    detected: bool
    bbox_conf: float = 0.0
    num_raw_detections: int = 0


class PitchKeypointDetector:
    """
    Thin wrapper around a YOLO pose model that returns one set of pitch keypoints.
    """

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        image_size: int = 640,
        num_keypoints: int = 32,
    ):
        self.model = YOLO(weights)
        self.device = device
        self.image_size = image_size
        self.num_keypoints = num_keypoints

    def _empty_result(
        self,
        *,
        bbox_conf: float = 0.0,
        num_raw_detections: int = 0,
    ) -> PitchKeypointResult:
        return PitchKeypointResult(
            xy=np.zeros((self.num_keypoints, 2), dtype=np.float32),
            conf=np.zeros((self.num_keypoints,), dtype=np.float32),
            detected=False,
            bbox_conf=bbox_conf,
            num_raw_detections=num_raw_detections,
        )

    def detect(self, frame: np.ndarray) -> PitchKeypointResult:
        result = self.model(
            frame,
            device=self.device,
            imgsz=self.image_size,
            conf=0.01,
            verbose=False,
        )[0]

        bbox_conf = 0.0
        box_count = 0
        det_scores = None
        if result.boxes is not None and len(result.boxes) > 0:
            box_count = int(len(result.boxes))
        if (
            result.boxes is not None
            and result.boxes.conf is not None
            and len(result.boxes.conf) > 0
        ):
            det_scores = np.asarray(result.boxes.conf.cpu().numpy(), dtype=np.float32)

        keypoint_count = 0
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoint_count = int(len(result.keypoints))

        num_raw_detections = box_count if box_count > 0 else keypoint_count

        det_idx = 0
        if keypoint_count == 0 or result.keypoints is None:
            if det_scores is not None and det_scores.size > 0:
                bbox_conf = float(det_scores[int(np.argmax(det_scores))])
            return self._empty_result(
                bbox_conf=bbox_conf,
                num_raw_detections=num_raw_detections,
            )

        if det_scores is not None and det_scores.size > 0:
            aligned_count = min(int(det_scores.size), keypoint_count)
            if aligned_count > 0:
                det_idx = int(np.argmax(det_scores[:aligned_count]))
                bbox_conf = float(det_scores[det_idx])

        keypoints_xy = result.keypoints.xy.cpu().numpy()[det_idx].astype(np.float32)
        if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
            keypoints_conf = (
                result.keypoints.conf.cpu().numpy()[det_idx].astype(np.float32)
            )
        else:
            keypoints_conf = np.ones((keypoints_xy.shape[0],), dtype=np.float32)

        if keypoints_xy.shape[0] != self.num_keypoints:
            # Keep output shape stable for downstream logic.
            padded_xy = np.zeros((self.num_keypoints, 2), dtype=np.float32)
            padded_conf = np.zeros((self.num_keypoints,), dtype=np.float32)
            n = min(self.num_keypoints, keypoints_xy.shape[0])
            padded_xy[:n] = keypoints_xy[:n]
            padded_conf[:n] = keypoints_conf[:n]
            keypoints_xy = padded_xy
            keypoints_conf = padded_conf

        return PitchKeypointResult(
            xy=keypoints_xy,
            conf=keypoints_conf,
            detected=True,
            bbox_conf=bbox_conf,
            num_raw_detections=num_raw_detections,
        )
