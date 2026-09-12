import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Train the CNN model."""

    def __init__(self,model: nn.Module) -> None:
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)

    def train(
        self,
        data_loader: DataLoader,
        epochs: int = 1,
    ) -> list[float]:

        loss_history = []
        self.model.train()
        for epoch in range(epochs):

            total_loss = 0.0
            for images, labels in data_loader:

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs,labels,)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            average_loss = (total_loss / len(data_loader))
            loss_history.append(average_loss)

        return loss_history