from app.ui.inference_controller import InferenceController


IMAGE_PATH = "data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg"


def main() -> None:

    controller = InferenceController()

    result = controller.predict(
        IMAGE_PATH
    )

    print(
        f"Prediction: {result}"
    )


if __name__ == "__main__":
    main()