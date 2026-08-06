import cv2
import numpy as np


class ImagePreprocessor:
    """Image preprocessing operations."""

    def resize(
        self,
        image: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:

        return cv2.resize(image, (width, height))