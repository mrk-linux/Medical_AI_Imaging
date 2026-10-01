import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Trainer:
    """Train and validate the CNN model."""

    def __init__(self, model: nn.Module) -> None:

        # Configure model
        self.model = model

        # Define loss function
        self.loss_function = nn.CrossEntropyLoss()

        # Configure optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
        )

        # Store best validation accuracy
        self.best_accuracy = 0.0


    def train(
        self,
        data_loader: DataLoader,
        epochs: int = 1,
    ) -> tuple[list[float], list[float]]:

        # Store training metrics
        loss_history = []
        accuracy_history = []

        # Set model to training mode
        self.model.train()

        for _ in range(epochs):

            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in data_loader:

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss = self.loss_function(outputs, labels)

                # Backward pass
                loss.backward()

                # Update model weights
                self.optimizer.step()

                total_loss += loss.item()

                # Calculate training accuracy
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # Save epoch metrics
            average_loss = total_loss / len(data_loader)
            accuracy = correct / total

            loss_history.append(average_loss)
            accuracy_history.append(accuracy)

        return loss_history, accuracy_history


    def validate(
        self,
        data_loader: DataLoader,
    ) -> tuple[float, float]:

        # Set model to evaluation mode
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient calculation during validation
        with torch.no_grad():

            for images, labels in data_loader:

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss = self.loss_function(outputs, labels)

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        average_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return average_loss, accuracy


    def is_best_model(self, accuracy: float) -> bool:

        # Check if current model is better than previous checkpoint
        if accuracy > self.best_accuracy:

            self.best_accuracy = accuracy
            return True

        return False