from Medical_AI_Imaging.app.input.image_loader import ImageLoader
from Medical_AI_Imaging.app.input.image_viewer import ImageViewer
from Medical_AI_Imaging.app.preprocessing.image_preprocessor import ImagePreprocessor


def __init__(self) -> None:
    self.loader = ImageLoader()
    self.viewer = ImageViewer()
    self.preprocessor = ImagePreprocessor()