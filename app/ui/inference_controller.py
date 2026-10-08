from app.input.inference_preprocessor import ImagePreprocessor
from app.models.inference import Inference
from app.models.model_loader import ModelLoader


class InferenceController:
    """Connect UI with AI inference pipeline."""

    def __init__(self) -> None:
        self.preprocessor = ImagePreprocessor()

        loader = ModelLoader()
        model = loader.load()

        self.inference = Inference(model)

    def predict(
        self,
        image_path: str,
    ) -> tuple[str, float]:
        """Run prediction for image."""

        image_tensor = self.preprocessor.process(
            image_path
        )

        prediction, confidence = self.inference.predict(
            image_tensor
        )

        return (
            prediction,
            confidence,
        )