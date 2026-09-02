import numpy as np

from app.preprocessing.normalizer import ImageNormalizer


def main() -> None:
    normalizer = ImageNormalizer()

    image = np.array(
        [[0, 128, 255]],
        dtype=np.uint8,
    )

    normalized_image = normalizer.normalize(image)

    print(normalized_image)
    print(normalized_image.dtype)


if __name__ == "__main__":
    main()