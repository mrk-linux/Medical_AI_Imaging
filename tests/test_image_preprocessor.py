import torch
from app.input.image_preprocessor import ImagePreprocessor


def main() -> None:
    preprocessor = ImagePreprocessor()

    image_path = (
        "data/raw/chest_xray/validation/NORMAL/IM-0128-0001.jpeg"
    )

    tensor = preprocessor.process(image_path)

    print(f"Tensor type: {type(tensor)}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Pixel range: {tensor.min()} - {tensor.max()}")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 512, 512)


if __name__ == "__main__":
    main()