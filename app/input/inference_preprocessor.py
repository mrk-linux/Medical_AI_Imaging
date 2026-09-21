import numpy as np
import torch
from PIL import Image


class ImagePreprocessor:
    """Prepare images for CNN model input."""

    def __init__(self,image_size: tuple[int, int] = (512, 512)) -> None:
        self.image_size = image_size

    def process(self,image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("L")

        # Resize image to match model input size.
        image = image.resize(self.image_size)
        image_array = np.array(image,dtype=np.float32)

        # Normalize pixel values to range [0, 1].
        image_array = image_array / 255.0
        tensor = torch.tensor(image_array)

        # Add batch and channel dimensions for CNN input.
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor