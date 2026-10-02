from pathlib import Path
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from src.fingerprint.config import PREPROCESSED_DIR
from src.fingerprint.dataset import split_fingerprint_dataset, FingerprintDataset
from experiments.fingerprint.embedding_model import FingerprintEmbeddingModel
from experiments.iris.metric_loss import BatchHardTripletLoss


PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

CHECKPOINT_DIR = PROJECT_ROOT / "models"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 7


class FingerprintSubjectBatchSampler(Sampler):

    def __init__(
        self,
        files,
        batch_size=16,
        images_per_subject=2
    ):

        self.batch_size = batch_size
        self.images_per_subject = images_per_subject

        subject_to_indices = defaultdict(list)

        for index, file in enumerate(files):

            subject = int(
                file.parent.name
            )

            subject_to_indices[
                subject
            ].append(index)

        self.subject_to_indices = (
            subject_to_indices
        )

    def __iter__(self):

        subjects = list(
            self.subject_to_indices.keys()
        )

        generator = torch.Generator()
        generator.manual_seed(
            torch.initial_seed()
        )

        permutation = torch.randperm(
            len(subjects),
            generator=generator
        ).tolist()

        subjects = [
            subjects[i]
            for i in permutation
        ]

        batch = []

        for subject in subjects:

            indices = (
                self.subject_to_indices[
                    subject
                ]
            )

            permutation = torch.randperm(
                len(indices),
                generator=generator
            ).tolist()

            selected = [
                indices[i]
                for i in permutation[
                    :self.images_per_subject
                ]
            ]

            batch.extend(selected)

            if len(batch) >= self.batch_size:

                yield batch[:self.batch_size]

                batch = batch[
                    self.batch_size:
                ]

        if batch:
            yield batch

    def __len__(self):

        total = sum(
            len(indices)
            for indices
            in self.subject_to_indices.values()
        )

        return total // self.batch_size


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    train_files, val_files, test_files = (
        split_fingerprint_dataset(
            PREPROCESSED_DIR
        )
    )

    all_files = (
        train_files
        + val_files
        + test_files
    )

    subjects = sorted({
        int(file.parent.name)
        for file in all_files
    })

    subject_to_label = {
        subject: index
        for index, subject in enumerate(subjects)
    }

    train_dataset = FingerprintDataset(
        train_files,
        subject_to_label,
        training=True
    )

    train_sampler = FingerprintSubjectBatchSampler(
        train_files,
        batch_size=BATCH_SIZE,
        images_per_subject=2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler
    )

    model = FingerprintEmbeddingModel(
        embedding_dim=128
    ).to(device)

    criterion = BatchHardTripletLoss(
        margin=0.3
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "fingerprint_embedding_best.pth"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            embeddings = model(images)

            loss = criterion(
                embeddings,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        average_loss = (
            total_loss / batches
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Triplet Loss: {average_loss:.4f}"
        )

        if average_loss < best_loss:

            best_loss = average_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "loss": average_loss,
                    "epoch": epoch
                },
                checkpoint_path
            )

            print(
                "  Best embedding model saved."
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    print(
        f"\nBest triplet loss: "
        f"{best_loss:.4f}"
    )


if __name__ == "__main__":
    main()
