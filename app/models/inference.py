import torch
import torch.nn as nn
import torch.nn.functional as F


class Inference:
    """Run inference with the CNN model."""

    LABELS = {
        0: "NORMAL",
        1: "PNEUMONIA",
    }

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        self.model = model

        # Switch model to evaluation mode for prediction.
        self.model.eval()

    def predict(
        self,
        image: torch.Tensor,
    ) -> tuple[str, float]:
        """Predict class and confidence."""

        with torch.no_grad():

            output = self.model(image)

            probabilities = F.softmax(
                output,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        label = self.LABELS[
            prediction.item()
        ]

        confidence_percentage = (
            confidence.item() * 100
        )

        return (
            label,
            confidence_percentage,
        )