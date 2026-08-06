from input.image_loader import ImageLoader
from input.image_viewer import ImageViewer


class Application:
    """Main application controller."""

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.viewer = ImageViewer()

    def run(self) -> None:
        image = self.loader.load(
            "data/images/chest_xray_001.jpg"
        )

        self.viewer.show(image)