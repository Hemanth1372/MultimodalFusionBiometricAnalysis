from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from .embedding_model import IrisEmbeddingModel
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
    / "iris_embedding_best.pth"
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

    _, _, test_files = split_iris_dataset(
        NORMALIZED_DIR
    )

    model = IrisEmbeddingModel(
        embedding_dim=128
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    train_files, _, _ = split_iris_dataset(
        NORMALIZED_DIR
    )

    train_embeddings = []
    train_labels = []

    with torch.no_grad():

        for file in train_files:

            image = load_image(file).to(device)

            embedding = model(image)

            train_embeddings.append(
                embedding
            )

            train_labels.append(
                int(file.name.split("_")[0])
            )

    train_embeddings = torch.cat(
        train_embeddings
    )

    correct = 0

    mistakes = []

    with torch.no_grad():

        for file in test_files:

            image = load_image(file).to(device)

            embedding = model(image)

            similarities = F.cosine_similarity(
                embedding,
                train_embeddings
            )

            best_index = torch.argmax(
                similarities
            ).item()

            true_label = int(
                file.name.split("_")[0]
            )

            predicted_label = (
                train_labels[best_index]
            )

            if predicted_label == true_label:

                correct += 1

            else:

                mistakes.append(
                    (
                        file.name,
                        true_label,
                        predicted_label,
                        similarities[best_index].item()
                    )
                )

    accuracy = correct / len(test_files)

    print(
        f"Test Accuracy: {accuracy:.4f}"
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
            f"{filename} | "
            f"True={true_label:02d} | "
            f"Pred={predicted_label:02d} | "
            f"Similarity={similarity:.4f}"
        )


if __name__ == "__main__":
    main()