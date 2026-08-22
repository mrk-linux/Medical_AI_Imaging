from input.dataset_splitter import DatasetSplitter


def main() -> None:
    splitter = DatasetSplitter()

    splitter.split(
        source_directory="data/raw/chest_xray/train",
        validation_directory="data/raw/chest_xray/validation",
        validation_ratio=0.2,
        random_seed=42,
    )

    print("Validation set created successfully.")


if __name__ == "__main__":
    main()