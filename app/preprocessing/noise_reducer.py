import cv2
import numpy as np


class NoiseReducer:
    """Reduce image noise using Gaussian blur."""

    def reduce(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            image,
            (5, 5),
            0,
        )