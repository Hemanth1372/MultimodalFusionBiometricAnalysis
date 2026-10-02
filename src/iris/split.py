from pathlib import Path
import random


def split_iris_dataset(
    root_dir,
    train_per_subject=6,
    val_per_subject=2,
    test_per_subject=2,
    seed=42
):

    root_dir = Path(root_dir)

    subject_files = {}

    for file in sorted(root_dir.glob("*.png")):

        subject = int(
            file.name.split("_")[0]
        )

        subject_files.setdefault(
            subject,
            []
        ).append(file)

    rng = random.Random(seed)

    train_files = []
    val_files = []
    test_files = []

    for subject in sorted(subject_files):

        files = subject_files[subject]

        if len(files) != (
            train_per_subject
            + val_per_subject
            + test_per_subject
        ):
            raise ValueError(
                f"Subject {subject} has "
                f"{len(files)} images"
            )

        files = files.copy()
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