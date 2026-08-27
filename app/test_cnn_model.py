import torch

from models.cnn_model import CNNModel
from models.trainer import Trainer


def main() -> None:
    model = CNNModel()
    trainer = Trainer(model)

    sample = torch.randn(1, 1, 512, 512)
    target = torch.tensor([0])

    output = model(sample)

    loss = trainer.loss_function(
        output,
        target,
    )

    first_parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    trainer.optimizer.zero_grad()

    loss.backward()

    trainer.optimizer.step()

    first_parameter_after = (
        next(model.parameters())
        .detach()
        .clone()
    )

    parameters_changed = not torch.equal(
        first_parameter_before,
        first_parameter_after,
    )

    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters changed: {parameters_changed}")


if __name__ == "__main__":
    main()