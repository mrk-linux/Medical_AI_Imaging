import torch

from app.input.dataset_loader import DatasetLoader
from app.input.dataset import Dataset
from app.models.cnn_model import CNNModel
from app.models.trainer import Trainer


def main() -> None:
    # Load real dataset samples
    loader = DatasetLoader()

    samples = loader.load(
        "data/raw/chest_xray/train"
    )

    if not samples:
        raise RuntimeError("No dataset samples were loaded.")

    # Prepare one real sample
    dataset = Dataset()

    prepared_samples = dataset.prepare(
        samples[:1]
    )

    image, label = prepared_samples[0]

    # Convert NumPy image to PyTorch tensor
    tensor = torch.from_numpy(image)

    # Add channel dimension
    tensor = tensor.unsqueeze(0)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Build model and trainer
    model = CNNModel()
    trainer = Trainer(model)

    # Forward pass
    output = model(tensor)

    # Prepare target
    target = torch.tensor([label])

    # Calculate loss
    loss = trainer.loss_function(
        output,
        target,
    )

    # Backpropagation
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    print(f"Original image shape: {image.shape}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Label: {label}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()