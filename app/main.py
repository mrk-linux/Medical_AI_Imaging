from input.image_loader import ImageLoader
from input.image_viewer import ImageViewer


def main() -> None:
    loader = ImageLoader()
    viewer = ImageViewer()

    image = loader.load("data/images/test.jpg")

    viewer.show(image)


if __name__ == "__main__":
    main()