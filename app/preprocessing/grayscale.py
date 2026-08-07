import cv2
import numpy as np


class GrayscaleConverter:
    """Convert images to grayscale."""

    def convert(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )