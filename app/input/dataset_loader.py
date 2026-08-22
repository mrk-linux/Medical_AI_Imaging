from pathlib import Path

import cv2
import numpy as np


class DatasetLoader:
    """Load labeled images from a dataset directory."""

    def load(self, directory: str) -> list[tuple[np.ndarray, str]]:
        dataset = []

        dataset_path = Path(directory)

        for label_directory in dataset_path.iterdir():
            if not label_directory.is_dir():
                continue

            label = label_directory.name

            for image_path in label_directory.iterdir():
                if not image_path.is_file():
                    continue

                image = cv2.imread(str(image_path))

                if image is None:
                    continue

                dataset.append((image, label))

        return dataset