import cv2
import numpy as np


class HistogramEqualizer:
    """Improve image contrast using histogram equalization."""

    def equalize(self, image: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(image)