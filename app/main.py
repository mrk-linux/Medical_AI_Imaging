from input.image_loader import ImageLoader


def main() -> None:
    loader = ImageLoader()
    image = loader.load("data/images/test.jpg")

    print(image.shape)


if __name__ == "__main__":
    main()