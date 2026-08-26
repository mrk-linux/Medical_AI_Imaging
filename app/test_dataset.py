from input.dataset import Dataset


def main() -> None:
    dataset = Dataset()

    normal_label = dataset.encode_label("NORMAL")
    pneumonia_label = dataset.encode_label("PNEUMONIA")

    print(f"NORMAL: {normal_label}")
    print(f"PNEUMONIA: {pneumonia_label}")


if __name__ == "__main__":
    main()