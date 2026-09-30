from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    conf: float
    cls_name: str  # class name
    cls_id: int  # class id
    team_id: Optional[int] = None
    team_conf: Optional[float] = None
    pitch_x_cm: Optional[float] = None
    pitch_y_cm: Optional[float] = None


@dataclass
class Track:
    track_id: int
    detection: Detection
    kit_feat_ema: Optional[np.ndarray] = None  # For smoothing

    @property
    def bbox(self) -> BBox:
        return (
            self.detection.bbox
        )  # to call it like tracked_det.bbox instead of tracked_det.det.bbox

    @property
    def conf(self) -> float:
        return self.detection.conf

    @property
    def cls_id(self) -> int:
        return self.detection.cls_id

    @property
    def cls_name(self) -> str:
        return self.detection.cls_name

    @property
    def team_id(self) -> Optional[int]:
        return self.detection.team_id

    @team_id.setter
    def team_id(self, value: int):
        object.__setattr__(self.detection, "team_id", value)

    @property
    def team_conf(self) -> Optional[float]:
        return self.detection.team_conf

    @team_conf.setter
    def team_conf(self, value: float):
        object.__setattr__(self.detection, "team_conf", value)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_position(self) -> Tuple[float, float]:
        # assume the bottom of the bbox is the foot position
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

    @property
    def pitch_position_cm(self) -> Optional[Tuple[float, float]]:
        if self.detection.pitch_x_cm is None or self.detection.pitch_y_cm is None:
            return None
        return (self.detection.pitch_x_cm, self.detection.pitch_y_cm)

    @property
    def pitch_x_cm(self) -> Optional[float]:
        return self.detection.pitch_x_cm

    @pitch_x_cm.setter
    def pitch_x_cm(self, value: Optional[float]):
        object.__setattr__(self.detection, "pitch_x_cm", value)

    @property
    def pitch_y_cm(self) -> Optional[float]:
        return self.detection.pitch_y_cm

    @pitch_y_cm.setter
    def pitch_y_cm(self, value: Optional[float]):
        object.__setattr__(self.detection, "pitch_y_cm", value)

    # Backward-compatible aliases for older meter-suffixed naming.
    @property
    def pitch_position_m(self) -> Optional[Tuple[float, float]]:
        return self.pitch_position_cm

    @property
    def pitch_x_m(self) -> Optional[float]:
        return self.pitch_x_cm

    @pitch_x_m.setter
    def pitch_x_m(self, value: Optional[float]):
        self.pitch_x_cm = value

    @property
    def pitch_y_m(self) -> Optional[float]:
        return self.pitch_y_cm

    @pitch_y_m.setter
    def pitch_y_m(self, value: Optional[float]):
        self.pitch_y_cm = value


@dataclass
class FrameData:
    frame_idx: int
    tracks: List[Track]
    width: int
    height: int
    img: Optional[np.ndarray] = None
