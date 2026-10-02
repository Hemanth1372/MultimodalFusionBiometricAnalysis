from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class IrisDataset(Dataset):

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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,
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
            file.name.split("_")[0]
        )

        label = self.subject_to_label[
            subject
        ]

        return image, label


def create_iris_dataloaders(
    train_files,
    val_files,
    test_files,
    batch_size=16
):

    all_files = (
        train_files
        + val_files
        + test_files
    )

    subjects = sorted(
        {
            int(file.name.split("_")[0])
            for file in all_files
        }
    )

    subject_to_label = {
        subject: index
        for index, subject in enumerate(subjects)
    }

    train_dataset = IrisDataset(
        train_files,
        subject_to_label,
        training=True
    )

    val_dataset = IrisDataset(
        val_files,
        subject_to_label,
        training=False
    )

    test_dataset = IrisDataset(
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