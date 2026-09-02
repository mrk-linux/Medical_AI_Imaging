import torch

from app.models.cnn_model import CNNModel
from app.models.model_saver import ModelSaver


def main() -> None:
    model = CNNModel()
    saver = ModelSaver()

    sample = torch.randn(1, 1, 512, 512)

    output_before = model(sample)

    path = "data/processed/cnn_model.pth"

    saver.save(model, path)

    loaded_model = CNNModel()
    saver.load(loaded_model, path)

    output_after = loaded_model(sample)

    outputs_match = torch.equal(
        output_before,
        output_after,
    )

    print(f"Model saved: {path}")
    print(f"Outputs match: {outputs_match}")


if __name__ == "__main__":
    main()