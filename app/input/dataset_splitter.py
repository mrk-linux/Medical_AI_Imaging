from pathlib import Path
import random
import shutil


class DatasetSplitter:
    """Split training data into train and validation sets."""

    def split(
        self,
        source_directory: str,
        validation_directory: str,
        validation_ratio: float = 0.2,
    ) -> None:
        source_path = Path(source_directory)
        validation_path = Path(validation_directory)

        for label_directory in source_path.iterdir():
            if not label_directory.is_dir():
                continue

            images = [
                image
                for image in label_directory.iterdir()
                if image.is_file()
            ]

            random.shuffle(images)

            validation_count = int(len(images) * validation_ratio)
            validation_images = images[:validation_count]

            target_label_directory = (
                validation_path / label_directory.name
            )
            target_label_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            for image_path in validation_images:
                shutil.move(
                    str(image_path),
                    str(target_label_directory / image_path.name),
                )