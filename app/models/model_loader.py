from pathlib import Path

from app.models.cnn_model import CNNModel
from app.models.model_saver import ModelSaver


class ModelLoader:
    """Load trained CNN model."""

    def __init__(
        self,
        model_path: str = "trained_models/cnn_model.pth",
    ) -> None:
        self.model_path = model_path

    def load(self) -> CNNModel:
        """Create model and load saved weights."""

        model = CNNModel()
        saver = ModelSaver()

        path = Path(self.model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        model = saver.load(
            model,
            self.model_path,
        )

        model.eval()

        return model