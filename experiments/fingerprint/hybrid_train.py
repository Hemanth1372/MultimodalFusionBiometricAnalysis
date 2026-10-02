from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fingerprint.config import PREPROCESSED_DIR
from src.fingerprint.dataset import (
    split_fingerprint_dataset,
    FingerprintDataset
)
from experiments.fingerprint.hybrid_model import (
    FingerprintHybridModel
)
from experiments.iris.metric_loss import (
    BatchHardTripletLoss
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 7

TRIPLET_WEIGHT = 0.5


class SubjectBatchSampler(Sampler):

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

        permutation = torch.randperm(
            len(subjects)
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
                len(indices)
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


def evaluate_classifier(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            _, logits = model(images)

            predictions = (
                logits.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def main():

    print(
        f"Device: {DEVICE}"
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

    val_dataset = FingerprintDataset(
        val_files,
        subject_to_label,
        training=False
    )

    train_sampler = SubjectBatchSampler(
        train_files,
        batch_size=BATCH_SIZE,
        images_per_subject=2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = FingerprintHybridModel(
        num_classes=45,
        embedding_dim=128
    ).to(DEVICE)

    classification_loss = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    triplet_loss = BatchHardTripletLoss(
        margin=0.3
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    checkpoint_path = (
        PROJECT_ROOT
        / "models"
        / "fingerprint_hybrid_best.pth"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        total_classification = 0.0
        total_triplet = 0.0
        batches = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            embeddings, logits = model(
                images
            )

            ce_loss = classification_loss(
                logits,
                labels
            )

            metric_loss = triplet_loss(
                embeddings,
                labels
            )

            loss = (
                ce_loss
                + TRIPLET_WEIGHT * metric_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_classification += (
                ce_loss.item()
            )
            total_triplet += (
                metric_loss.item()
            )

            batches += 1

        average_loss = (
            total_loss / batches
        )

        average_ce = (
            total_classification
            / batches
        )

        average_triplet = (
            total_triplet
            / batches
        )

        val_accuracy = evaluate_classifier(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"CE: {average_ce:.4f} | "
            f"Triplet: {average_triplet:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                },
                checkpoint_path
            )

            print(
                "  Best hybrid model saved."
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

    print()
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()

