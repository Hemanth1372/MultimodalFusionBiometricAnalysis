from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 7


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu"
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:
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

        total += images.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return average_loss, accuracy


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
        split_iris_dataset(
            NORMALIZED_DIR
        )
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_iris_dataloaders(
        train_files,
        val_files,
        test_files,
        batch_size=BATCH_SIZE
    )

    model = IrisResNet18(
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

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "iris_resnet18_best.pth"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_accuracy = (
            run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device
            )
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
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                },
                checkpoint_path
            )

            print(
                "  Best model saved."
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
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_accuracy = (
        run_epoch(
            model,
            test_loader,
            criterion,
            None,
            device
        )
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()