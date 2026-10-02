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

    train_files, _, test_files = (
        split_iris_dataset(
            NORMALIZED_DIR
        )
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

    subject_embeddings = {}

    with torch.no_grad():

        for file in train_files:

            subject = int(
                file.name.split("_")[0]
            )

            image = load_image(file).to(device)

            embedding = model(image)

            subject_embeddings.setdefault(
                subject,
                []
            ).append(embedding)

    print("Test embedding analysis:\n")

    errors = 0

    with torch.no_grad():

        for file in test_files:

            true_subject = int(
                file.name.split("_")[0]
            )

            image = load_image(file).to(device)

            query = model(image)

            similarities = {}

            for subject, embeddings in (
                subject_embeddings.items()
            ):

                subject_matrix = torch.cat(
                    embeddings
                )

                scores = F.cosine_similarity(
                    query,
                    subject_matrix
                )

                similarities[subject] = (
                    scores.mean().item()
                )

            true_similarity = similarities[
                true_subject
            ]

            predicted_subject = max(
                similarities,
                key=similarities.get
            )

            predicted_similarity = (
                similarities[
                    predicted_subject
                ]
            )

            if predicted_subject != true_subject:

                errors += 1

                print(
                    f"{file.name:<38} "
                    f"True={true_subject:02d} "
                    f"Pred={predicted_subject:02d} "
                    f"TrueSim={true_similarity:.4f} "
                    f"PredSim={predicted_similarity:.4f}"
                )

    print(
        f"\nTotal errors: {errors}/{len(test_files)}"
    )


if __name__ == "__main__":
    main()