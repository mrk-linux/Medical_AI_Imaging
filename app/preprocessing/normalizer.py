import numpy as np


class ImageNormalizer:
    """Normalize image pixel values."""

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0