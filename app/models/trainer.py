import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Trainer:
    """Train and validate the CNN model."""

    def __init__(self,model: nn.Module,) -> None:
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
        )

    def train(self,data_loader: DataLoader,epochs: int = 1,) -> list[float]:

        loss_history = []
        self.model.train()

        for _ in range(epochs):

            total_loss = 0.0

            for images, labels in data_loader:

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            loss_history.append(
                total_loss / len(data_loader)
            )

        return loss_history

    def validate(self,data_loader: DataLoader,) -> tuple[float, float]:

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient calculation because weights are not updated.
        with torch.no_grad():

            for images, labels in data_loader:

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        average_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return average_loss, accuracy