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

    # Create model and trainer
    model = CNNModel()
    trainer = Trainer(model)

    # Train model
    loss_history = trainer.train(data_loader,epochs=1,)
    print(f"Training loss: {loss_history}")

    # Validate model
    validation_loss, accuracy = trainer.validate(data_loader)
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

    # Save trained model
    saver = ModelSaver()
    saver.save(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()