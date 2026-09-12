import numpy as np
import torch

from app.input.pytorch_dataset import PyTorchDataset
from app.input.pytorch_data_loader import PyTorchDataLoader


def main() -> None:
    samples = [
        (np.zeros((512, 512), dtype=np.float32),0),
        (np.ones((512, 512), dtype=np.float32),1),
        (np.full((512, 512), 0.5, dtype=np.float32),0),
        (np.full((512, 512), 0.8, dtype=np.float32),1),
    ]

    dataset = PyTorchDataset(samples)

    data_loader_creator = PyTorchDataLoader(dataset,batch_size=2,shuffle=False,)

    data_loader = data_loader_creator.create()

    images, labels = next(iter(data_loader))

    print(f"Images type: {type(images)}")
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")

    print(f"Labels type: {type(labels)}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()