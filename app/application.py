from input.image_loader import ImageLoader
from input.image_viewer import ImageViewer
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.normalizer import ImageNormalizer


class Application:
    """Main application controller."""

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.viewer = ImageViewer()
        self.preprocessor = ImagePreprocessor()
        self.grayscale_converter = GrayscaleConverter()
        self.normalizer = ImageNormalizer()

    def run(self) -> None:
        image = self.loader.load(
            "data/images/chest_xray_002.jpg"
        )

        image = self.preprocessor.resize(
            image,
            width=512,
            height=512,
        )

        image = self.grayscale_converter.convert(image)

        self.viewer.show(image)