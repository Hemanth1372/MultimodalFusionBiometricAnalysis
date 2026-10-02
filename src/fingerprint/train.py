from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from .config import PREPROCESSED_DIR, PROJECT_ROOT
from .dataset import create_fingerprint_dataloaders
from .model import FingerprintResNet18


BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 7

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fingerprint_resnet18_best.pth"


def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_fingerprint_dataloaders(
        PREPROCESSED_DIR,
        batch_size=BATCH_SIZE
    )

    model = FingerprintResNet18(
        num_classes=45
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print("Early stopping.")
            break

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    train()