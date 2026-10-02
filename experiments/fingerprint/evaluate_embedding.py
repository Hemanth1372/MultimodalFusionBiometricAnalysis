from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fingerprint.config import PREPROCESSED_DIR
from src.fingerprint.dataset import (
    split_fingerprint_dataset,
    FingerprintDataset
)
from experiments.fingerprint.embedding_model import (
    FingerprintEmbeddingModel
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def extract_embeddings(model, loader):

    embeddings = []
    labels = []

    model.eval()

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(DEVICE)

            batch_embeddings = model(
                images
            )

            embeddings.append(
                batch_embeddings.cpu()
            )

            labels.append(
                batch_labels
            )

    return (
        torch.cat(embeddings),
        torch.cat(labels)
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

    model = FingerprintEmbeddingModel(
        embedding_dim=128
    ).to(DEVICE)

    checkpoint = torch.load(
        PROJECT_ROOT
        / "models"
        / "fingerprint_embedding_best.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    embeddings, labels = extract_embeddings(
        model,
        test_loader
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1
    )

    similarity = (
        embeddings @ embeddings.T
    )

    correct = 0
    total = len(embeddings)

    for i in range(total):

        scores = similarity[i].clone()

        scores[i] = -1

        nearest = scores.argmax().item()

        if labels[nearest] == labels[i]:
            correct += 1

    accuracy = correct / total

    print(
        f"Test samples: {total}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Nearest-neighbor accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Correct: {correct}/{total}"
    )


if __name__ == "__main__":
    main()
