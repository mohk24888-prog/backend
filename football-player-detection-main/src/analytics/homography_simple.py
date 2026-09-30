from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt


class HomographyTransformer:
    def __init__(
        self,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.H, _ = cv2.findHomography(source, target, cv2.RANSAC, 150)
        if self.H is None:
            raise ValueError("Homography not found.")

    def transform_points(
        self, points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.H)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
        self, image: npt.NDArray[np.uint8], resolution: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        if len(image.shape) not in (2, 3):
            raise ValueError("Image must be 2D or 3D.")
        return cv2.warpPerspective(image, self.H, resolution)
