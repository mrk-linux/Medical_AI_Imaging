import torch
from app.models.cnn_model import CNNModel
from app.models.inference import Inference


def main() -> None:
    model = CNNModel()
    inference = Inference(model)
    image = torch.randn(1,1,512,512)
    prediction = inference.predict(image)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()