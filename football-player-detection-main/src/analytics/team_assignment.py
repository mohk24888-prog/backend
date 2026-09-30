import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict
from src.core.types import Track


class TeamAssigner:
    """
    Assigns team IDs to players using robust color features and K-means clustering.
    Implements per-player color extraction (median LAB), grass masking,
    canonical team ID assignment, and temporal smoothing.
    """

    def __init__(self, ema_alpha: float = 0.9):
        self.team_centroids: Optional[np.ndarray] = (
            None  # (2, 2) for 2 teams, 2 features (a, b)
        )
        self.kmeans: Optional[KMeans] = None
        self.ema_alpha = ema_alpha  # Smoothing factor (higher = more sticky)
        self.player_features_buffer: List[
            np.ndarray
        ] = []  # Collect features for fitting

        # Track smoothing state
        # Map track_id -> smoothed_feature (np.array)
        # We can also store this on the Track object directly if it persists,
        # but storing here ensures persistence across frames if Track objects are recreated.
        self.track_smoothing: Dict[int, np.ndarray] = {}

    def crop_torso(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Crop the upper body (torso) to focus on the jersey.
        Takes the central 50% of the torso crop to minimize background.
        """
        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]

        # Clamp bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        box_width = x2 - x1
        box_height = y2 - y1

        # Crop torso: 20% -> 65% vertically, 20% -> 80% horizontally
        cx1 = x1 + int(box_width * 0.2)
        cx2 = x1 + int(box_width * 0.8)
        cy1 = y1 + int(box_height * 0.2)
        cy2 = y1 + int(box_height * 0.65)

        # Refine crop to central 50% of the above to avoid edges/background
        tc_w = cx2 - cx1
        tc_h = cy2 - cy1

        # Take middle 50%
        sx1 = cx1 + int(tc_w * 0.25)
        sx2 = cx2 - int(tc_w * 0.25)
        sy1 = cy1 + int(tc_h * 0.25)
        sy2 = cy2 - int(tc_h * 0.25)

        # Clamp again
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(W, sx2), min(H, sy2)

        if sx2 <= sx1 or sy2 <= sy1:
            return None

        return frame[sy1:sy2, sx1:sx2]

    def get_clustering_features(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract robust clustering features from a player crop.
        Uses Median LAB 'a' and 'b' channels with grass masking.
        """
        crop = self.crop_torso(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        # Masking Grass
        # Convert to HSV for easy green extraction
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Green range (approx 35-90 Hue for grass)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_crop, lower_green, upper_green)

        # Invert mask: we want NON-green pixels
        non_grass_mask = cv2.bitwise_not(mask)

        # If too many pixels are masked out (e.g. > 90%), fallback to full crop
        if cv2.countNonZero(non_grass_mask) < (crop.shape[0] * crop.shape[1] * 0.1):
            non_grass_mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255

        # Feature Extraction using LAB
        # Use only non-grass pixels
        lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        # Reshape to list of pixels
        lab_pixels = lab_crop[non_grass_mask > 0].reshape(-1, 3)

        if len(lab_pixels) == 0:
            return None

        # Compute Robust Statistics: Median
        # We perform clustering on 'a' and 'b' channels (indices 1 and 2)
        # Lightness 'L' (index 0) is less reliable for team color separation
        median_lab = np.median(lab_pixels, axis=0)

        return median_lab[1:]  # Return [a, b]

    def fit(self, tracks: List[Track], frame: np.ndarray):
        """
        Collect features from valid tracks.
        Call this for the first N frames to build the dataset.
        """
        for track in tracks:
            # Skip goalkeeper or ref if possible, or assume they are outliers
            # For now, just collect Player classes
            if track.cls_name != "Player":
                continue

            feat = self.get_clustering_features(frame, track.bbox)
            if feat is not None:
                self.player_features_buffer.append(feat)

    def train_model(self):
        """
        Train the K-Means model on collected features.
        Canonicalize centroids to prevent label flipping.
        """
        if len(self.player_features_buffer) < 10:
            # Not enough data
            print("Warning: Not enough player features to train team assignment.")
            # Create dummy centroids to avoid crash?
            return

        X = np.array(self.player_features_buffer)

        # Train K-Means
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        self.kmeans.fit(X)

        centroids = self.kmeans.cluster_centers_

        # Canonicalize: Team 0 should have the lower 'a' value (or some determinstic rule)
        # This prevents labels from flipping between runs.
        # Centroids shape: (2, 2) -> [[a1, b1], [a2, b2]]
        if centroids[0][0] > centroids[1][0]:
            # Swap centroids so Team 0 is smaller 'a'
            self.team_centroids = np.array([centroids[1], centroids[0]])
            # We also need to swap the labels in the fitted kmeans if we were using it for prediction directly
            # But we will use nearest neighbor to self.team_centroids manually.
        else:
            self.team_centroids = centroids

        print(f"Team Assignment Trained. Centroids (a,b): {self.team_centroids}")

    def resolve_goalkeepers_team_id(
        self,
        players: List[Track],
        goalkeepers: List[Track],
    ) -> np.ndarray:
        """
        Resolve goalkeeper team IDs from player position centroids.

        Players are expected to already have team_id in {0, 1}.
        If only one team centroid is available, all goalkeepers are assigned to it.
        If no team centroid can be computed, returns -1 for each goalkeeper.
        """
        if len(goalkeepers) == 0:
            return np.zeros((0,), dtype=np.int32)

        valid_players = [p for p in players if p.team_id in (0, 1)]
        if len(valid_players) == 0:
            return np.full((len(goalkeepers),), -1, dtype=np.int32)

        players_xy = np.asarray(
            [p.foot_position for p in valid_players], dtype=np.float32
        )
        players_team_id = np.asarray(
            [int(p.team_id) for p in valid_players], dtype=np.int32
        )
        goalkeepers_xy = np.asarray(
            [g.foot_position for g in goalkeepers], dtype=np.float32
        )

        team_centroids: Dict[int, np.ndarray] = {}
        for team_id in (0, 1):
            team_xy = players_xy[players_team_id == team_id]
            if len(team_xy) > 0:
                team_centroids[team_id] = team_xy.mean(axis=0)

        if len(team_centroids) == 0:
            return np.full((len(goalkeepers),), -1, dtype=np.int32)
        if len(team_centroids) == 1:
            only_team_id = next(iter(team_centroids))
            return np.full((len(goalkeepers),), only_team_id, dtype=np.int32)

        goalkeepers_team_id: List[int] = []
        team_0_centroid = team_centroids[0]
        team_1_centroid = team_centroids[1]
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        return np.asarray(goalkeepers_team_id, dtype=np.int32)

    def predict(self, frame: np.ndarray, tracks: List[Track]):
        """
        Assign team IDs to the tracks in the frame.
        """
        if self.team_centroids is None:
            return

        for track in tracks:
            if track.cls_name != "Player":
                continue

            # 1. Extract Raw Feature
            raw_feat = self.get_clustering_features(frame, track.bbox)
            if raw_feat is None:
                continue

            # 2. Smoothing (EMA)
            track_id = track.track_id
            if track_id in self.track_smoothing:
                # Update EMA
                smoothed_feat = (
                    self.ema_alpha * self.track_smoothing[track_id]
                    + (1 - self.ema_alpha) * raw_feat
                )
                self.track_smoothing[track_id] = smoothed_feat
            else:
                # Initialize
                smoothed_feat = raw_feat
                self.track_smoothing[track_id] = smoothed_feat

            # 3. Assign Team
            # Manually find closest centroid
            # dists = [norm(feat - c0), norm(feat - c1)]
            dists = np.linalg.norm(self.team_centroids - smoothed_feat, axis=1)
            team_id = np.argmin(dists)

            # Compute confidence (margin)
            # margin = |d0 - d1| / (d0 + d1)
            # 0 = uncertain, 1 = very certain
            margin = abs(dists[0] - dists[1]) / (dists[0] + dists[1] + 1e-6)

            # Assign to track (mutating the Track/Detection object)
            track.team_id = int(team_id)
            track.team_conf = float(margin)

        players_with_team = [
            t for t in tracks if t.cls_name == "Player" and t.team_id in (0, 1)
        ]
        goalkeepers = [t for t in tracks if t.cls_name == "Goalkeeper"]
        goalkeepers_team_id = self.resolve_goalkeepers_team_id(
            players=players_with_team,
            goalkeepers=goalkeepers,
        )
        for goalkeeper, team_id in zip(goalkeepers, goalkeepers_team_id):
            if int(team_id) < 0:
                goalkeeper.team_id = None
                goalkeeper.team_conf = None
                continue
            goalkeeper.team_id = int(team_id)
            goalkeeper.team_conf = 1.0
