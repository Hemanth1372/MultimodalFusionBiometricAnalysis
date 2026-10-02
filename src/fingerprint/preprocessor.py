from pathlib import Path

import cv2
import numpy as np

from .config import (
    DATASET,
    PREPROCESSED_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)


def preprocess_fingerprint(
    image_path: Path,
    output_path: Path
):
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read {image_path}"
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    enhanced = cv2.addWeighted(
        image,
        1.5,
        cv2.GaussianBlur(
            image,
            (0, 0),
            2
        ),
        -0.5,
        0
    )

    enhanced = cv2.resize(
        enhanced,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_CUBIC
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        enhanced
    )


def preprocess_dataset():

    files = sorted(
        DATASET.glob(
            r"*\Fingerprint\*.BMP"
        )
    )

    for image_path in files:

        subject = image_path.parent.parent.name

        output_path = (
            PREPROCESSED_DIR
            / subject
            / f"{image_path.stem}.png"
        )

        preprocess_fingerprint(
            image_path,
            output_path
        )

    return len(files)


if __name__ == "__main__":

    count = preprocess_dataset()

    print(
        f"Processed: {count}"
    )
