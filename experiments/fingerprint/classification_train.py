from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

from src.fingerprint.config import PREPROCESSED_DIR
from src.fingerprint.dataset import (
    split_fingerprint_dataset,
    FingerprintDataset
)


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

NUM_CLASSES = 45
EMBEDDING_DIM = 128

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 40
PATIENCE = 7


# ============================================================
# MODEL
# ============================================================


class FingerprintClassificationModel(
    nn.Module
):

    def __init__(
        self,
        embedding_dim=128,
        num_classes=45
    ):

        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        old_conv = self.backbone.conv1

        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():

            new_conv.weight[:] = (
                old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

        self.backbone.conv1 = new_conv

        # Remove ResNet's original 1000-class classifier.
        # This makes the backbone output 512 features.
        num_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.embedding = nn.Linear(
            num_features,
            embedding_dim
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        features = self.backbone(
            x
        )

        embedding = self.embedding(
            features
        )

        logits = self.classifier(
            embedding
        )

        return logits, embedding


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            logits, _ = model(
                images
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        correct / total
        if total
        else 0.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create fixed dataset split
    # --------------------------------------------------------

    (
        train_files,
        val_files,
        test_files
    ) = split_fingerprint_dataset(
        PREPROCESSED_DIR
    )

    print(
        "\n========== DATASET =========="
    )

    print(
        f"Train: {len(train_files)}"
    )

    print(
        f"Validation: {len(val_files)}"
    )

    print(
        f"Test: {len(test_files)}"
    )

    # --------------------------------------------------------
    # Create subject labels
    # --------------------------------------------------------

    all_files = (
        train_files
        + val_files
        + test_files
    )

    subjects = sorted({
        int(file.parent.name)
        for file in all_files
    })

    if len(subjects) != NUM_CLASSES:

        raise ValueError(
            f"Expected {NUM_CLASSES} subjects, "
            f"found {len(subjects)}"
        )

    subject_to_label = {
        subject: index
        for index, subject in enumerate(
            subjects
        )
    }

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

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

    test_dataset = FingerprintDataset(
        test_files,
        subject_to_label,
        training=False
    )

    # --------------------------------------------------------
    # Create dataloaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = FingerprintClassificationModel(
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "fingerprint_classification_best.pth"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits, _ = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            batches += 1

        average_loss = (
            total_loss / batches
        )

        val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best model using validation accuracy
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "epoch":
                        epoch,

                    "embedding_dim":
                        EMBEDDING_DIM,

                    "num_classes":
                        NUM_CLASSES
                },
                checkpoint_path
            )

            print(
                "  Best classification model saved."
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

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_accuracy = evaluate(
        model,
        test_loader,
        device
    )

    print(
        "\n========== FINAL RESULT =========="
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        "\nSaved checkpoint:"
    )

    print(
        checkpoint_path
    )


if __name__ == "__main__":
    main()

