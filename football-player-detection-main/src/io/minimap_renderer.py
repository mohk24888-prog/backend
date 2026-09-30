from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.types import Track


class MinimapRenderer:
    """
    Draws a 2D pitch minimap and overlays projected track positions.
    """

    def __init__(
        self,
        pitch_length_m: float,
        pitch_width_m: float,
        minimap_width_px: int = 360,
        minimap_height_px: int = 232,
        margin_px: int = 20,
        units_per_meter: float = 1.0,
    ):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.minimap_width_px = minimap_width_px
        self.minimap_height_px = minimap_height_px
        self.margin_px = margin_px
        self.units_per_meter = units_per_meter

        self._base_canvas = self._build_pitch_canvas()

    def _m_to_units(self, value_m: float) -> float:
        return value_m * self.units_per_meter

    def _world_to_canvas(self, x_m: float, y_m: float) -> Tuple[int, int]:
        x = int(
            np.clip(
                (x_m / self.pitch_length_m) * (self.minimap_width_px - 1),
                0,
                self.minimap_width_px - 1,
            )
        )
        y = int(
            np.clip(
                (y_m / self.pitch_width_m) * (self.minimap_height_px - 1),
                0,
                self.minimap_height_px - 1,
            )
        )
        return x, y

    def _draw_line_m(
        self,
        canvas: np.ndarray,
        p1_m: Tuple[float, float],
        p2_m: Tuple[float, float],
        color,
        thickness=1,
    ):
        p1 = self._world_to_canvas(*p1_m)
        p2 = self._world_to_canvas(*p2_m)
        cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    def _draw_rect_m(
        self,
        canvas: np.ndarray,
        left_m: float,
        top_m: float,
        right_m: float,
        bottom_m: float,
        color,
        thickness=1,
    ):
        p1 = self._world_to_canvas(left_m, top_m)
        p2 = self._world_to_canvas(right_m, bottom_m)
        cv2.rectangle(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    def _build_pitch_canvas(self) -> np.ndarray:
        canvas = np.full(
            (self.minimap_height_px, self.minimap_width_px, 3),
            (38, 130, 56),
            dtype=np.uint8,
        )
        line_color = (240, 240, 240)

        # Outer boundary.
        self._draw_rect_m(
            canvas,
            0.0,
            0.0,
            self.pitch_length_m,
            self.pitch_width_m,
            line_color,
            thickness=2,
        )

        # Halfway line.
        self._draw_line_m(
            canvas,
            (self.pitch_length_m / 2.0, 0.0),
            (self.pitch_length_m / 2.0, self.pitch_width_m),
            line_color,
            thickness=2,
        )

        # Center circle and spot.
        center = self._world_to_canvas(
            self.pitch_length_m / 2.0, self.pitch_width_m / 2.0
        )
        radius_px = int(
            (self._m_to_units(9.15) / self.pitch_length_m) * (self.minimap_width_px - 1)
        )
        cv2.circle(
            canvas, center, max(radius_px, 1), line_color, 1, lineType=cv2.LINE_AA
        )
        cv2.circle(canvas, center, 2, line_color, -1, lineType=cv2.LINE_AA)

        # Penalty boxes.
        self._draw_rect_m(
            canvas,
            0.0,
            self._m_to_units(13.84),
            self._m_to_units(16.5),
            self._m_to_units(54.16),
            line_color,
            thickness=1,
        )
        self._draw_rect_m(
            canvas,
            self.pitch_length_m - self._m_to_units(16.5),
            self._m_to_units(13.84),
            self.pitch_length_m,
            self._m_to_units(54.16),
            line_color,
            thickness=1,
        )

        # Goal boxes.
        self._draw_rect_m(
            canvas,
            0.0,
            self._m_to_units(24.84),
            self._m_to_units(5.5),
            self._m_to_units(43.16),
            line_color,
            thickness=1,
        )
        self._draw_rect_m(
            canvas,
            self.pitch_length_m - self._m_to_units(5.5),
            self._m_to_units(24.84),
            self.pitch_length_m,
            self._m_to_units(43.16),
            line_color,
            thickness=1,
        )

        # Penalty spots.
        left_pen = self._world_to_canvas(
            self._m_to_units(11.0), self.pitch_width_m / 2.0
        )
        right_pen = self._world_to_canvas(
            self.pitch_length_m - self._m_to_units(11.0), self.pitch_width_m / 2.0
        )
        cv2.circle(canvas, left_pen, 2, line_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, right_pen, 2, line_color, -1, lineType=cv2.LINE_AA)

        return canvas

    def _team_arrow_color(self, team_id: Optional[int]) -> Tuple[int, int, int]:
        if team_id == 0:
            return (100, 100, 255)
        if team_id == 1:
            return (255, 100, 100)
        return (200, 200, 200)

    def _draw_pass_arrows(
        self,
        canvas: np.ndarray,
        pass_arrows: List[Tuple[Tuple[float, float], Tuple[float, float], int, float]],
        canvas_w: int,
        canvas_h: int,
    ) -> None:
        """Draw fading pass arrows on a canvas.

        Each arrow is (passer_pos_cm, receiver_pos_cm, team_id, age_fraction)
        where age_fraction goes from 0.0 (fresh) to 1.0 (about to expire).
        """
        for passer_pos, receiver_pos, team_id, age_frac in pass_arrows:
            p1x = int(
                np.clip(
                    (passer_pos[0] / self.pitch_length_m) * (canvas_w - 1),
                    0,
                    canvas_w - 1,
                )
            )
            p1y = int(
                np.clip(
                    (passer_pos[1] / self.pitch_width_m) * (canvas_h - 1),
                    0,
                    canvas_h - 1,
                )
            )
            p2x = int(
                np.clip(
                    (receiver_pos[0] / self.pitch_length_m) * (canvas_w - 1),
                    0,
                    canvas_w - 1,
                )
            )
            p2y = int(
                np.clip(
                    (receiver_pos[1] / self.pitch_width_m) * (canvas_h - 1),
                    0,
                    canvas_h - 1,
                )
            )

            base_color = self._team_arrow_color(team_id)
            alpha = 1.0 - age_frac
            color = tuple(int(c * alpha) for c in base_color)
            thickness = 2 if age_frac < 0.5 else 1

            cv2.arrowedLine(
                canvas,
                (p1x, p1y),
                (p2x, p2y),
                color,
                thickness,
                line_type=cv2.LINE_AA,
                tipLength=0.25,
            )

    def _track_color(self, track: Track) -> Tuple[int, int, int]:
        if track.cls_name == "Ball":
            return (245, 245, 245)
        if track.cls_name == "Referee":
            return (0, 255, 255)
        if track.team_id is None:
            return (200, 200, 200)
        if track.team_id == 0:
            return (0, 0, 255)
        return (255, 0, 0)

    def draw_minimap(
        self,
        tracks: List[Track],
        status_text: Optional[str] = None,
        possessor_id: Optional[int] = None,
        pass_arrows: Optional[
            List[Tuple[Tuple[float, float], Tuple[float, float], int, float]]
        ] = None,
    ) -> np.ndarray:
        canvas = self._base_canvas.copy()

        for track in tracks:
            if track.pitch_x_cm is None or track.pitch_y_cm is None:
                continue

            x_px, y_px = self._world_to_canvas(track.pitch_x_cm, track.pitch_y_cm)
            color = self._track_color(track)

            if track.cls_name == "Ball":
                cv2.circle(canvas, (x_px, y_px), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (x_px, y_px), 3, color, -1, lineType=cv2.LINE_AA)
            else:
                if possessor_id is not None and track.track_id == possessor_id:
                    cv2.circle(
                        canvas, (x_px, y_px), 7, (0, 255, 255), 2, lineType=cv2.LINE_AA
                    )
                cv2.circle(canvas, (x_px, y_px), 4, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (x_px, y_px), 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        if pass_arrows:
            self._draw_pass_arrows(
                canvas, pass_arrows, self.minimap_width_px, self.minimap_height_px
            )

        if status_text:
            cv2.putText(
                canvas,
                status_text,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (250, 250, 250),
                1,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def overlay(
        self, frame: np.ndarray, tracks: List[Track], status_text: Optional[str] = None
    ) -> np.ndarray:
        minimap = self.draw_minimap(tracks=tracks, status_text=status_text)
        h_map, w_map = minimap.shape[:2]
        h_frame, w_frame = frame.shape[:2]

        x0 = max((w_frame - w_map) // 2, 0)
        y0 = max(h_frame - h_map - self.margin_px, 0)
        if x0 + w_map > w_frame:
            x0 = max(w_frame - w_map, 0)

        frame[y0 : y0 + h_map, x0 : x0 + w_map] = minimap
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + w_map, y0 + h_map),
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        return frame

    def render_split_panel(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        panel_height: int = 200,
        status_text: Optional[str] = None,
        ball_speed_kmh: Optional[float] = None,
        possessor_id: Optional[int] = None,
        pass_arrows: Optional[
            List[Tuple[Tuple[float, float], Tuple[float, float], int, float]]
        ] = None,
    ) -> np.ndarray:
        """Return a new image: video frame on top, dark panel with radar below."""
        h_frame, w_frame = frame.shape[:2]

        # Build a larger minimap that fits the panel with padding.
        pad = 12
        map_h = panel_height - 2 * pad
        # Pitch aspect ratio: length / width.
        pitch_aspect = self.pitch_length_m / self.pitch_width_m
        map_w = int(map_h * pitch_aspect)
        if map_w > w_frame - 2 * pad:
            map_w = w_frame - 2 * pad
            map_h = int(map_w / pitch_aspect)

        # Render pitch canvas at the panel size.
        old_w, old_h = self.minimap_width_px, self.minimap_height_px
        self.minimap_width_px = map_w
        self.minimap_height_px = map_h
        scaled_canvas = self._build_pitch_canvas()
        self.minimap_width_px = old_w
        self.minimap_height_px = old_h

        # Draw tracks onto scaled canvas.
        for track in tracks:
            if track.pitch_x_cm is None or track.pitch_y_cm is None:
                continue
            x_px = int(
                np.clip(
                    (track.pitch_x_cm / self.pitch_length_m) * (map_w - 1), 0, map_w - 1
                )
            )
            y_px = int(
                np.clip(
                    (track.pitch_y_cm / self.pitch_width_m) * (map_h - 1), 0, map_h - 1
                )
            )
            color = self._track_color(track)
            if track.cls_name == "Ball":
                cv2.circle(
                    scaled_canvas, (x_px, y_px), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA
                )
                cv2.circle(
                    scaled_canvas, (x_px, y_px), 4, color, -1, lineType=cv2.LINE_AA
                )
            else:
                if possessor_id is not None and track.track_id == possessor_id:
                    cv2.circle(
                        scaled_canvas,
                        (x_px, y_px),
                        8,
                        (0, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                cv2.circle(
                    scaled_canvas, (x_px, y_px), 5, color, -1, lineType=cv2.LINE_AA
                )
                cv2.circle(
                    scaled_canvas, (x_px, y_px), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA
                )

        if pass_arrows:
            self._draw_pass_arrows(scaled_canvas, pass_arrows, map_w, map_h)

        # Build the dark panel.
        panel = np.full((panel_height, w_frame, 3), (20, 20, 20), dtype=np.uint8)

        # Centre the pitch in the panel.
        x0 = (w_frame - map_w) // 2
        y0 = pad
        panel[y0 : y0 + map_h, x0 : x0 + map_w] = scaled_canvas
        cv2.rectangle(
            panel,
            (x0 - 1, y0 - 1),
            (x0 + map_w, y0 + map_h),
            (60, 60, 60),
            1,
            lineType=cv2.LINE_AA,
        )

        # Status text to the left of the pitch.
        if status_text:
            cv2.putText(
                panel,
                status_text,
                (12, panel_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                lineType=cv2.LINE_AA,
            )

        # Ball speed below status text.
        if ball_speed_kmh is not None:
            speed_label = f"Ball: {ball_speed_kmh:.0f} km/h"
            cv2.putText(
                panel,
                speed_label,
                (12, panel_height // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        # Vertical stack: frame on top, panel below.
        return np.vstack([frame, panel])
