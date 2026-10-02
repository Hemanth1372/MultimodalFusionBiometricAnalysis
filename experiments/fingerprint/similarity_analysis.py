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

    genuine = []
    impostor = []

    for i in range(len(labels)):

        for j in range(i + 1, len(labels)):

            score = similarity[i, j].item()

            if labels[i] == labels[j]:
                genuine.append(score)
            else:
                impostor.append(score)

    genuine_tensor = torch.tensor(
        genuine
    )

    impostor_tensor = torch.tensor(
        impostor
    )

    print(
        f"Genuine pairs: "
        f"{len(genuine)}"
    )

    print(
        f"Impostor pairs: "
        f"{len(impostor)}"
    )

    print()

    print(
        f"Genuine mean similarity: "
        f"{genuine_tensor.mean():.4f}"
    )

    print(
        f"Genuine min similarity: "
        f"{genuine_tensor.min():.4f}"
    )

    print(
        f"Genuine max similarity: "
        f"{genuine_tensor.max():.4f}"
    )

    print()

    print(
        f"Impostor mean similarity: "
        f"{impostor_tensor.mean():.4f}"
    )

    print(
        f"Impostor min similarity: "
        f"{impostor_tensor.min():.4f}"
    )

    print(
        f"Impostor max similarity: "
        f"{impostor_tensor.max():.4f}"
    )


if __name__ == "__main__":
    main()

