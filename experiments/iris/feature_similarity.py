from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from .model import IrisResNet18
from .split import split_iris_dataset


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


def load_image(file):

    image = cv2.imread(
        str(file),
        cv2.IMREAD_GRAYSCALE
    )

    image = (
        torch.from_numpy(image)
        .float()
        / 255.0
    )

    return image.unsqueeze(0).unsqueeze(0)


def main():

    device = torch.device("cpu")

    train_files, _, test_files = (
        split_iris_dataset(
            NORMALIZED_DIR
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

    train_features = []
    train_labels = []

    test_features = []
    test_labels = []

    def extract_features(files, feature_list, label_list):

        for file in files:

            image = load_image(file).to(device)

            features = []

            def hook(module, inputs, output):
                features.append(
                    output.detach()
                )

            handle = (
                model.model.avgpool
                .register_forward_hook(hook)
            )

            with torch.no_grad():
                model(image)

            handle.remove()

            feature = features[0].flatten(
                start_dim=1
            )

            feature = F.normalize(
                feature,
                p=2,
                dim=1
            )

            feature_list.append(feature)

            label_list.append(
                int(file.name.split("_")[0])
            )

    extract_features(
        train_files,
        train_features,
        train_labels
    )

    extract_features(
        test_files,
        test_features,
        test_labels
    )

    train_features = torch.cat(
        train_features
    )

    test_features = torch.cat(
        test_features
    )

    similarities = torch.mm(
        test_features,
        train_features.t()
    )

    correct = 0

    mistakes = []

    for i in range(len(test_files)):

        best_index = torch.argmax(
            similarities[i]
        ).item()

        true_label = test_labels[i]

        predicted_label = (
            train_labels[best_index]
        )

        if predicted_label == true_label:

            correct += 1

        else:

            mistakes.append(
                (
                    test_files[i].name,
                    true_label,
                    predicted_label,
                    similarities[i][best_index].item()
                )
            )

    accuracy = correct / len(test_files)

    print(
        f"Nearest-neighbor accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Correct: {correct}/{len(test_files)}"
    )

    print("\nMistakes:")

    for (
        filename,
        true_label,
        predicted_label,
        similarity
    ) in mistakes:

        print(
            f"{filename:<38} "
            f"True={true_label:02d} "
            f"Pred={predicted_label:02d} "
            f"Similarity={similarity:.4f}"
        )


if __name__ == "__main__":
    main()