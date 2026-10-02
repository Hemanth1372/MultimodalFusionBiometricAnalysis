from pathlib import Path
import random

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FingerprintDataset(Dataset):

    def __init__(
        self,
        files,
        subject_to_label,
        training=False
    ):
        self.files = files
        self.subject_to_label = subject_to_label
        self.training = training

        self.augmentation = transforms.Compose([
            transforms.RandomRotation(
                degrees=8
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05)
            ),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file = self.files[index]

        image = cv2.imread(
            str(file),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read {file}"
            )

        image = torch.from_numpy(
            image
        ).float() / 255.0

        image = image.unsqueeze(0)

        if self.training:
            image = self.augmentation(image)

        subject = int(
            file.parent.name
        )

        label = self.subject_to_label[
            subject
        ]

        return image, label


def split_fingerprint_dataset(
    root_dir,
    train_per_subject=6,
    val_per_subject=2,
    test_per_subject=2,
    seed=42
):
    root_dir = Path(root_dir)

    subject_files = {}

    for subject_dir in sorted(
        root_dir.iterdir(),
        key=lambda x: int(x.name)
    ):
        if not subject_dir.is_dir():
            continue

        files = sorted(
            subject_dir.glob("*.png")
        )

        subject = int(subject_dir.name)

        if len(files) != (
            train_per_subject
            + val_per_subject
            + test_per_subject
        ):
            raise ValueError(
                f"Subject {subject} has "
                f"{len(files)} images"
            )

        subject_files[subject] = files

    rng = random.Random(seed)

    train_files = []
    val_files = []
    test_files = []

    for subject in sorted(subject_files):

        files = subject_files[subject].copy()

        rng.shuffle(files)

        train_files.extend(
            files[:train_per_subject]
        )

        val_files.extend(
            files[
                train_per_subject:
                train_per_subject + val_per_subject
            ]
        )

        test_files.extend(
            files[
                train_per_subject + val_per_subject:
            ]
        )

    return (
        train_files,
        val_files,
        test_files
    )


def create_fingerprint_dataloaders(
    root_dir,
    batch_size=16
):
    train_files, val_files, test_files = (
        split_fingerprint_dataset(root_dir)
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

    test_dataset = FingerprintDataset(
        test_files,
        subject_to_label,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )
