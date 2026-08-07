import numpy as np

from preprocessing.normalizer import ImageNormalizer


def main() -> None:
    image = np.array([[0, 128, 255]], dtype=np.uint8)

    normalizer = ImageNormalizer()
    normalized = normalizer.normalize(image)

    print(normalized)
    print(normalized.dtype)


if __name__ == "__main__":
    main()