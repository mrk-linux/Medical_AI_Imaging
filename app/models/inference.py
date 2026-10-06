import torch
import torch.nn as nn


class Inference:
    """Run inference with the CNN model."""

    LABELS = {
        0: "NORMAL",
        1: "PNEUMONIA",
    }

    def __init__(self,model: nn.Module) -> None:
        self.model = model

        # Switch model to evaluation mode for prediction.
        self.model.eval()

    def predict(self,image: torch.Tensor) -> str:

        # Disable gradient calculation because we only predict.
        with torch.no_grad():

            output = self.model(image)
            # Select the class with the highest score.
            prediction = torch.argmax(output, dim=1).item()

        return self.LABELS[prediction]