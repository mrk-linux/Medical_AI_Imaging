import torch
import torch.nn as nn


class Trainer:
    """Configure training components for the CNN model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
        )