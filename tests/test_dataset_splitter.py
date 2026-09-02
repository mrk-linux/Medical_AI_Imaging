from pathlib import Path
from tempfile import TemporaryDirectory

from app.input.dataset_splitter import DatasetSplitter


def create_test_dataset(base_path: Path) -> None:
    for label in ("NORMAL", "PNEUMONIA"):
        label_path = base_path / label
        label_path.mkdir(parents=True)

        for index in range(10):
            image_path = label_path / f"image_{index}.jpg"
            image_path.touch()


def main() -> None:
    with TemporaryDirectory() as temp_directory:
        root = Path(temp_directory)

        train_path = root / "train"
        validation_path = root / "validation"

        train_path.mkdir()

        create_test_dataset(train_path)

        splitter = DatasetSplitter()

        splitter.split(
            source_directory=str(train_path),
            validation_directory=str(validation_path),
            validation_ratio=0.2,
        )

        normal_train = len(
            list((train_path / "NORMAL").iterdir())
        )
        normal_validation = len(
            list((validation_path / "NORMAL").iterdir())
        )

        pneumonia_train = len(
            list((train_path / "PNEUMONIA").iterdir())
        )
        pneumonia_validation = len(
            list((validation_path / "PNEUMONIA").iterdir())
        )

        print(
            f"NORMAL -> train: {normal_train}, "
            f"validation: {normal_validation}"
        )

        print(
            f"PNEUMONIA -> train: {pneumonia_train}, "
            f"validation: {pneumonia_validation}"
        )


if __name__ == "__main__":
    main()