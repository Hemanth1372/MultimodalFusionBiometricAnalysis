from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fingerprint.config import PREPROCESSED_DIR
from src.fingerprint.dataset import (
    split_fingerprint_dataset,
    FingerprintDataset
)
from src.fingerprint.model import FingerprintResNet18


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    _, _, test_files = (
        split_fingerprint_dataset(
            PREPROCESSED_DIR
        )
    )

    subjects = sorted({
        int(file.parent.name)
        for file in test_files
    })

    subject_to_label = {
        subject: index
        for index, subject in enumerate(subjects)
    }

    label_to_subject = {
        label: subject
        for subject, label in subject_to_label.items()
    }

    test_dataset = FingerprintDataset(
        test_files,
        subject_to_label,
        training=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    model = FingerprintResNet18(
        num_classes=45
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            PROJECT_ROOT
            / "models"
            / "fingerprint_resnet18_best.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
                .cpu()
            )

            true_labels.extend(
                labels.tolist()
            )

            predicted_labels.extend(
                predictions.tolist()
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(subjects)))
    )

    print(
        f"Test Accuracy: {accuracy:.4f}"
    )

    correct = sum(
        t == p
        for t, p in zip(
            true_labels,
            predicted_labels
        )
    )

    print(
        f"Correct: {correct}/{len(true_labels)}"
    )

    print("Misclassified samples:")

    for file, true_label, predicted_label in zip(
        test_files,
        true_labels,
        predicted_labels
    ):

        if true_label != predicted_label:

            true_subject = label_to_subject[
                true_label
            ]

            predicted_subject = label_to_subject[
                predicted_label
            ]

            print(
                f"{file.name} | "
                f"True: {true_subject} | "
                f"Predicted: {predicted_subject}"
            )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Fingerprint Test Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Subject"
    )

    plt.ylabel(
        "True Subject"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(subjects)
    )

    plt.xticks(
        tick_positions,
        subjects,
        rotation=90
    )

    plt.yticks(
        tick_positions,
        subjects
    )

    plt.tight_layout()

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "fingerprint"
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    print()

    print(
        f"Saved: {output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()
