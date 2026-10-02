from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .split import split_iris_dataset
from .dataset import create_iris_dataloaders
from .model import IrisResNet18


PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "quality_check"
    / "normalized"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "iris_resnet18_best.pth"
)

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "quality_check"
    / "confusion_matrix.png"
)


def main():

    device = torch.device("cpu")

    train_files, val_files, test_files = (
        split_iris_dataset(NORMALIZED_DIR)
    )

    _, _, _, _, _, test_loader = (
        create_iris_dataloaders(
            train_files,
            val_files,
            test_files,
            batch_size=16
        )
    )

    model = IrisResNet18(num_classes=45).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            true_labels.extend(
                labels.numpy()
            )

            predicted_labels.extend(
                predictions.numpy()
            )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(45)
    )

    print("\nNon-zero confusion pairs:\n")

    for true_label in range(45):

        for predicted_label in range(45):

            if (
                true_label != predicted_label
                and matrix[true_label, predicted_label] > 0
            ):

                print(
                    f"Subject {true_label + 1:02d} "
                    f"-> Subject {predicted_label + 1:02d}: "
                    f"{matrix[true_label, predicted_label]}"
                )

    plt.figure(figsize=(12, 10))

    plt.imshow(matrix)

    plt.colorbar()

    plt.xlabel("Predicted Subject")
    plt.ylabel("True Subject")
    plt.title("Iris Recognition Confusion Matrix")

    plt.xticks(
        np.arange(45),
        np.arange(1, 46),
        rotation=90
    )

    plt.yticks(
        np.arange(45),
        np.arange(1, 46)
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT,
        dpi=200
    )

    plt.close()

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()