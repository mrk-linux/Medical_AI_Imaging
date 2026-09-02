from app.input.dataset_loader import DatasetLoader


def main() -> None:
    loader = DatasetLoader()

    dataset = loader.load(
        "data/raw/chest_xray/train"
    )

    print(f"Loaded samples: {len(dataset)}")

    if dataset:
        image, label = dataset[0]

        print(f"First image shape: {image.shape}")
        print(f"First label: {label}")


if __name__ == "__main__":
    main()