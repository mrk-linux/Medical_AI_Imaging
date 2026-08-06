from input.image_loader import ImageLoader
from input.image_viewer import ImageViewer
from preprocessing.image_preprocessor import ImagePreprocessor


class Application:
    """Main application controller."""

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.viewer = ImageViewer()
        self.preprocessor = ImagePreprocessor()
    def run(self) -> None:
        image = self.loader.load(
            "data/images/chest_xray_001.jpg"
        )

        image = self.preprocessor.resize(
            image,
            width=512,
            height=512,
        )

        self.viewer.show(image)