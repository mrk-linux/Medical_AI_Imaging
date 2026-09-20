from app.input.image_preprocessor import ImagePreprocessor
from app.models.cnn_model import CNNModel
from app.models.inference import Inference


def main() -> None:
    image_path = ("data/raw/chest_xray/validation/NORMAL/IM-0128-0001.jpeg")
    preprocessor = ImagePreprocessor()

    # Convert raw image into CNN input tensor.
    image_tensor = preprocessor.process(image_path)
    model = CNNModel()
    inference = Inference(model)

    # Run model prediction on processed image.
    prediction = inference.predict(image_tensor)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()