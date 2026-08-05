import cv2
import numpy as np


class ImageViewer:
    """Display images using OpenCV."""

    def show(self, image: np.ndarray, window_name: str = "Image") -> None:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()