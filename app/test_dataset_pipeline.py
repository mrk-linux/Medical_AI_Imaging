from input.dataset_loader import DatasetLoader
from input.dataset import Dataset


def main() -> None:
    loader = DatasetLoader()
    dataset = Dataset()

    samples = loader.load(
        "data/raw/chest_xray/train"
    )

    prepared_samples = dataset.prepare(
        samples[:5]
    )

    print(f"Loaded samples: {len(samples)}")
    print(
        f"Prepared samples: {len(prepared_samples)}"
    )

    image, label = prepared_samples[0]

    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min: {image.min()}")
    print(f"Image max: {image.max()}")
    print(f"Encoded label: {label}")


if __name__ == "__main__":
    main()