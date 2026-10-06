from pathlib import Path

from app.input.dataset_loader import DatasetLoader
from app.input.dataset import Dataset
from app.input.pytorch_dataset import PyTorchDataset
from app.input.pytorch_data_loader import PyTorchDataLoader
from app.models.cnn_model import CNNModel
from app.models.trainer import Trainer
from app.models.model_saver import ModelSaver


TRAIN_DATASET_PATH = "data/raw/chest_xray/train"
VALIDATION_DATASET_PATH = "data/raw/chest_xray/validation"
TEST_DATASET_PATH = "data/raw/chest_xray/test"
MODEL_PATH = "trained_models/cnn_model.pth"


def create_data_loader(path: str, shuffle: bool):

    # Load dataset samples
    loader = DatasetLoader()
    samples = loader.load(path)

    if not samples:
        raise RuntimeError(f"No dataset samples found: {path}")

    print(f"Loaded samples from {path}: {len(samples)}")

    # Prepare dataset
    dataset = Dataset()
    prepared_samples = dataset.prepare(samples)

    # Create PyTorch dataset
    pytorch_dataset = PyTorchDataset(prepared_samples)

    # Create DataLoader
    data_loader_creator = PyTorchDataLoader(
        pytorch_dataset,
        batch_size=32,
        shuffle=shuffle,
    )

    return data_loader_creator.create()


def main() -> None:

    # Create train, validation and test loaders
    train_loader = create_data_loader(
        TRAIN_DATASET_PATH,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        VALIDATION_DATASET_PATH,
        shuffle=False,
    )

    test_loader = create_data_loader(
        TEST_DATASET_PATH,
        shuffle=False,
    )

    # Create model and load checkpoint
    model = CNNModel()
    saver = ModelSaver()

    if Path(MODEL_PATH).exists():
        model = saver.load(model, MODEL_PATH)
        print(f"Loaded checkpoint: {MODEL_PATH}")

    # Create trainer
    trainer = Trainer(model)

    # Train model
    loss_history, accuracy_history = trainer.train(
        train_loader,
        epochs=1,
    )

    print(f"Training loss: {loss_history}")
    print(f"Training accuracy: {accuracy_history}")

    # Validate model
    validation_loss, validation_accuracy = trainer.validate(
        validation_loader
    )

    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")

    # Test model
    test_loss, test_accuracy = trainer.validate(test_loader)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save best model checkpoint
    if trainer.is_best_model(validation_accuracy):
        saver.save(model, MODEL_PATH)
        print(f"Best model saved: {MODEL_PATH}")
    else:
        print("Current model is not better than previous checkpoint.")


if __name__ == "__main__":
    main()