import numpy as np
import torch

from app.input.pytorch_dataset import PyTorchDataset
from app.input.pytorch_data_loader import PyTorchDataLoader
from app.models.cnn_model import CNNModel
from app.models.trainer import Trainer


def main() -> None:

    samples = [
        (
            np.zeros((512, 512), dtype=np.float32),
            0,
        ),
        (
            np.ones((512, 512), dtype=np.float32),
            1,
        ),
        (
            np.full((512, 512), 0.5, dtype=np.float32),
            0,
        ),
        (
            np.full((512, 512), 0.8, dtype=np.float32),
            1,
        ),
    ]

    dataset = PyTorchDataset(samples)

    data_loader_creator = PyTorchDataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    data_loader = data_loader_creator.create()

    model = CNNModel()

    trainer = Trainer(model)

    parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    loss, accuracy = trainer.validate(
        data_loader,
    )

    parameter_after = (
        next(model.parameters())
        .detach()
        .clone()
    )

    parameters_changed = not torch.equal(
        parameter_before,
        parameter_after,
    )

    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Parameters changed: {parameters_changed}")


if __name__ == "__main__":
    main()