from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.normalizer import ImageNormalizer


class Dataset:
    """Prepare dataset samples for AI model."""

    LABEL_MAP = {
        "NORMAL": 0,
        "PNEUMONIA": 1,
    }

    def __init__(self) -> None:
        self.preprocessor = ImagePreprocessor()
        self.grayscale_converter = GrayscaleConverter()
        self.normalizer = ImageNormalizer()

    def encode_label(self, label: str) -> int:
        return self.LABEL_MAP[label]

    def prepare(
        self,
        samples: list[tuple[object, str]],
    ) -> list[tuple[object, int]]:

        prepared_dataset = []

        for image, label in samples:

            # Resize image
            image = self.preprocessor.resize(
                image,
                width=512,
                height=512,
            )

            # Convert to grayscale
            image = self.grayscale_converter.convert(image)

            # Normalize pixel values
            image = self.normalizer.normalize(image)

            # Convert label to number
            encoded_label = self.encode_label(label)

            prepared_dataset.append((image,encoded_label,))

        return prepared_dataset