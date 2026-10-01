from pathlib import Path

from app.input.dataset_loader import DatasetLoader
from app.input.dataset import Dataset
from app.input.pytorch_dataset import PyTorchDataset
from app.input.pytorch_data_loader import PyTorchDataLoader
from app.models.cnn_model import CNNModel
from app.models.trainer import Trainer
from app.models.model_saver import ModelSaver


DATASET_PATH = "data/raw/chest_xray/train"
MODEL_PATH = "trained_models/cnn_model.pth"


def main() -> None:

    # Load raw dataset
    loader = DatasetLoader()
    samples = loader.load(DATASET_PATH)

    if not samples:
        raise RuntimeError("No dataset samples were loaded.")

    print(f"Loaded samples: {len(samples)}")

    # Prepare dataset
    dataset = Dataset()
    prepared_samples = dataset.prepare(samples)

    # Create PyTorch dataset
    pytorch_dataset = PyTorchDataset(prepared_samples)

    # Create DataLoader
    data_loader_creator = PyTorchDataLoader(
        pytorch_dataset,
        batch_size=32,
        shuffle=True,
    )

    data_loader = data_loader_creator.create()

    # Create model and load previous checkpoint
    model = CNNModel()
    saver = ModelSaver()

    if Path(MODEL_PATH).exists():
        model = saver.load(model, MODEL_PATH)
        print(f"Loaded checkpoint: {MODEL_PATH}")

    # Create trainer
    trainer = Trainer(model)

    # Train model
    loss_history, accuracy_history = trainer.train(
        data_loader,
        epochs=1,
    )

    print(f"Training loss: {loss_history}")
    print(f"Training accuracy: {accuracy_history}")

    # Validate model
    validation_loss, accuracy = trainer.validate(data_loader)

    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

    # Save best model checkpoint
    if trainer.is_best_model(accuracy):
        saver.save(model, MODEL_PATH)
        print(f"Best model saved: {MODEL_PATH}")
    else:
        print("Current model is not better than previous checkpoint.")


if __name__ == "__main__":
    main()