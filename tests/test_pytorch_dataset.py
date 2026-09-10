import numpy as np
import torch

from app.input.pytorch_dataset import PyTorchDataset


def main() -> None:
    samples = [
        (np.zeros((512, 512), dtype=np.float32),0,),
        (np.ones((512, 512), dtype=np.float32),1,),
    ]

    dataset = PyTorchDataset(samples)

    print(f"Dataset length: {len(dataset)}")

    image, label = dataset[0]

    print(f"Image type: {type(image)}")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")

    print(f"Label type: {type(label)}")
    print(f"Label value: {label}")
    print(f"Label dtype: {label.dtype}")


if __name__ == "__main__":
    main()