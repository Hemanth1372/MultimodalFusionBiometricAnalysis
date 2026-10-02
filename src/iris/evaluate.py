from pathlib import Path
from collections import defaultdict

import torch

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


def main():

    device = torch.device("cpu")

    train_files, val_files, test_files = (
        split_iris_dataset(
            NORMALIZED_DIR
        )
    )

    _, _, _, _, _, test_loader = (
        create_iris_dataloaders(
            train_files,
            val_files,
            test_files,
            batch_size=16
        )
    )

    model = IrisResNet18(
        num_classes=45
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    subject_stats = defaultdict(
        lambda: [0, 0]
    )

    mistakes = []

    test_files = list(test_files)

    file_index = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(
                images.to(device)
            )

            predictions = outputs.argmax(
                dim=1
            )

            for i in range(
                len(labels)
            ):

                true_label = labels[i].item()
                predicted_label = predictions[i].item()

                true_subject = (
                    true_label + 1
                )

                predicted_subject = (
                    predicted_label + 1
                )

                current_file = test_files[
                    file_index
                ]

                file_index += 1

                subject_stats[
                    true_subject
                ][1] += 1

                if true_label == predicted_label:

                    subject_stats[
                        true_subject
                    ][0] += 1

                else:

                    mistakes.append(
                        (
                            true_subject,
                            predicted_subject,
                            current_file.name
                        )
                    )

    correct = sum(
        values[0]
        for values in subject_stats.values()
    )

    total = sum(
        values[1]
        for values in subject_stats.values()
    )

    print(
        f"Test Accuracy: {correct / total:.4f}"
    )

    print(
        f"Correct: {correct}/{total}"
    )

    print("\nPer-subject accuracy:")

    for subject in sorted(
        subject_stats
    ):

        correct_count, total_count = (
            subject_stats[subject]
        )

        print(
            f"Subject {subject:02d}: "
            f"{correct_count}/{total_count}"
        )

    print("\nMistakes:")

    for (
        true_subject,
        predicted_subject,
        filename
    ) in mistakes:

        print(
            f"True Subject: {true_subject:02d} "
            f"-> Predicted: {predicted_subject:02d} "
            f"| File: {filename}"
        )


if __name__ == "__main__":
    main()