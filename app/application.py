from app.input.image_loader import ImageLoader
from app.input.image_viewer import ImageViewer
from app.preprocessing.image_preprocessor import ImagePreprocessor
from app.preprocessing.grayscale import GrayscaleConverter
from app.preprocessing.normalizer import ImageNormalizer
from app.preprocessing.histogram_equalizer import HistogramEqualizer
from app.preprocessing.noise_reducer import NoiseReducer

class Application:
    """Main application controller."""

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.viewer = ImageViewer()
        self.preprocessor = ImagePreprocessor()
        self.grayscale_converter = GrayscaleConverter()
        self.normalizer = ImageNormalizer()
        self.histogram_equalizer = HistogramEqualizer()
        self.noise_reducer = NoiseReducer()

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
        image = self.noise_reducer.reduce(image)
        image = self.histogram_equalizer.equalize(image)
        self.viewer.show(image)