from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fingerprint.config import PREPROCESSED_DIR


def main():

    subjects = ["1", "2", "3", "4", "5"]

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(10, 14)
    )

    for row, subject in enumerate(subjects):

        files = sorted(
            (PREPROCESSED_DIR / subject).glob("*.png")
        )

        for col, file in enumerate(files[:2]):

            image = cv2.imread(
                str(file),
                cv2.IMREAD_GRAYSCALE
            )

            axes[row, col].imshow(
                image,
                cmap="gray"
            )

            axes[row, col].set_title(
                file.stem
            )

            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()