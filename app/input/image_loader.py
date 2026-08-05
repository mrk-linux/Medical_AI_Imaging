from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    """Load images from disk."""

    def load(self, image_path: str) -> np.ndarray:
        image = cv2.imread(str(Path(image_path)))

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return image