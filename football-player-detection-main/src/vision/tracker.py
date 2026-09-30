from ultralytics import YOLO
from typing import Dict, Iterator, List
from src.core.types import Detection, Track, FrameData


class Tracker:
    """
    Tracker class for tracking objects in a video.
    """

    def __init__(
        self,
        weights: str,
        device: str = "mps",
        image_size: int = 640,
        tracker_cfg: str = "configs/botsort.yaml",
        iou_thresh: float = 0.6,
        conf_ball: float = 0.15,
        conf_player: float = 0.25,
    ):
        self.model = YOLO(weights)
        self.names: Dict[int, str] = self.model.names

        self.device = device
        self.image_size = image_size
        self.tracker_cfg = tracker_cfg
        self.iou_thresh = iou_thresh
        self.conf_ball = conf_ball
        self.conf_player = conf_player

    def track_video(
        self,
        video_path: str,
        output_path: str | None = None,
        run_name: str | None = None,
        vid_stride: int = 1,
        save: bool = False,
    ) -> Iterator:
        """
        Track video using the model.

        Args:
            video_path: Path to the video file.
            output_path: Path to the output directory when ``save`` is True.
            run_name: Name of the run when ``save`` is True.
            vid_stride: Video stride.
            save: Whether to persist Ultralytics tracking outputs.

        Returns:
            Iterator of results.
        """
        base_conf = min(self.conf_ball, self.conf_player)
        track_kwargs = {
            "source": video_path,
            "device": self.device,
            "imgsz": self.image_size,
            "conf": base_conf,
            "iou": self.iou_thresh,
            "tracker": self.tracker_cfg,
            "vid_stride": vid_stride,
            "save": save,
            "stream": True,
            "persist": True,
            "verbose": False,
        }
        if save:
            if not output_path:
                raise ValueError("output_path is required when save=True")
            track_kwargs["project"] = output_path
            track_kwargs["name"] = run_name or "tracking"
            track_kwargs["exist_ok"] = True
        return self.model.track(**track_kwargs)

    def to_tracks(self, result: Iterator) -> List[Track]:
        """
        Convert the result to a list of tracks.

        Args:
            result: Result from the model.

        Returns:
            List of tracks.
        """

        # check if the result has boxes and if it is empty return an empty list
        if result.boxes is None or len(result.boxes) == 0:
            return []

        # assign the boxes to a variable and get the xyxy, cls, conf
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        # get the track ids if available
        ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)

        # create a list of tracks
        tracks: List[Track] = []

        # loop through the boxes and create a track for each box
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cls_id = int(cls[i])
            p = float(conf[i])
            cls_name = self.names.get(cls_id, str(cls_id))

            # set the confidence threshold based on the class name
            if cls_name == "Ball" and p < self.conf_ball:
                continue
            if cls_name != "Ball" and p < self.conf_player:
                continue

            track_id = int(ids[i]) if ids is not None else -1

            # create a detection object
            det = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                conf=p,
                cls_name=cls_name,
                cls_id=cls_id,
            )
            tracks.append(Track(track_id=track_id, detection=det))

        return tracks

    def frames(
        self,
        video_path: str,
        output_path: str | None = None,
        run_name: str | None = None,
        vid_stride: int = 1,
        save: bool = False,
    ) -> Iterator[FrameData]:
        """
        Track video and yield frame-level tracking results.

        Args:
            video_path: Path to the video file.
            output_path: Path to the output directory when ``save`` is True.
            run_name: Name of the run when ``save`` is True.
            vid_stride: Video stride.
            save: Whether to persist Ultralytics tracking outputs.

        Returns:
            Iterator of FrameData.
        """
        results = self.track_video(
            video_path=video_path,
            output_path=output_path,
            run_name=run_name,
            vid_stride=vid_stride,
            save=save,
        )
        for frame_idx, r in enumerate(results, start=1):
            tracks = self.to_tracks(r)
            h, w = r.orig_img.shape[:2]
            yield FrameData(
                frame_idx=frame_idx, tracks=tracks, width=w, height=h, img=r.orig_img
            )
