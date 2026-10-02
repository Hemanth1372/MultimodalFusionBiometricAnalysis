from pathlib import Path
import random


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

IRIS_TRAIN_PER_EYE = 3
IRIS_VAL_PER_EYE = 1
IRIS_TEST_PER_EYE = 1

FINGER_TRAIN_PER_FINGER = 3
FINGER_VAL_PER_FINGER = 1
FINGER_TEST_PER_FINGER = 1


# ============================================================
# GENERIC SPLIT
# ============================================================

def split_files(
    files,
    train_count,
    val_count,
    test_count,
    rng
):
    files = sorted(files)

    expected = (
        train_count
        + val_count
        + test_count
    )

    if len(files) != expected:
        raise ValueError(
            f"Expected {expected} files, "
            f"found {len(files)}."
        )

    files = files.copy()
    rng.shuffle(files)

    train = files[:train_count]

    val = files[
        train_count:
        train_count + val_count
    ]

    test = files[
        train_count + val_count:
    ]

    return train, val, test


# ============================================================
# IRIS SPLIT
# ============================================================

def split_iris(root_dir):
    root_dir = Path(root_dir)
    rng = random.Random(SEED)

    subject_files = {}

    for file in sorted(
        root_dir.glob("*.png")
    ):

        parts = file.stem.split("_")

        subject = int(parts[0])

        subject_files.setdefault(
            subject,
            []
        ).append(file)

    train = []
    val = []
    test = []

    for subject in sorted(subject_files):

        files = subject_files[subject]

        left_files = [
            file
            for file in files
            if "_left_" in file.name
        ]

        right_files = [
            file
            for file in files
            if "_right_" in file.name
        ]

        if len(left_files) != 5:
            raise ValueError(
                f"Subject {subject} has "
                f"{len(left_files)} left-eye images. "
                f"Expected 5."
            )

        if len(right_files) != 5:
            raise ValueError(
                f"Subject {subject} has "
                f"{len(right_files)} right-eye images. "
                f"Expected 5."
            )

        left_train, left_val, left_test = split_files(
            left_files,
            IRIS_TRAIN_PER_EYE,
            IRIS_VAL_PER_EYE,
            IRIS_TEST_PER_EYE,
            rng
        )

        right_train, right_val, right_test = split_files(
            right_files,
            IRIS_TRAIN_PER_EYE,
            IRIS_VAL_PER_EYE,
            IRIS_TEST_PER_EYE,
            rng
        )

        train.extend(
            left_train + right_train
        )

        val.extend(
            left_val + right_val
        )

        test.extend(
            left_test + right_test
        )

    return train, val, test



# ============================================================
# FINGERPRINT SPLIT
# ============================================================

def get_finger_name(filename):
    name = filename.lower()

    fingers = [
        "index",
        "little",
        "middle",
        "ring",
        "thumb"
    ]

    for finger in fingers:
        if finger in name:
            return finger

    raise ValueError(
        f"Cannot identify finger "
        f"from filename: {filename}"
    )


def get_hand(filename):
    name = filename.lower()

    if "left" in name:
        return "left"

    if "right" in name:
        return "right"

    raise ValueError(
        f"Cannot identify hand "
        f"from filename: {filename}"
    )


def split_fingerprint(root_dir):
    root_dir = Path(root_dir)
    rng = random.Random(SEED)

    train = []
    val = []
    test = []

    subject_dirs = sorted(
        root_dir.iterdir(),
        key=lambda x: int(x.name)
    )

    for subject_dir in subject_dirs:

        if not subject_dir.is_dir():
            continue

        subject = int(
            subject_dir.name
        )

        files = sorted(
            subject_dir.glob("*.png")
        )

        if len(files) != 10:
            raise ValueError(
                f"Subject {subject} has "
                f"{len(files)} fingerprint samples. "
                f"Expected 10."
            )

        files = files.copy()

        rng.shuffle(files)

        train.extend(
            files[:6]
        )

        val.extend(
            files[6:8]
        )

        test.extend(
            files[8:10]
        )

    return train, val, test


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_iris_summary(
    train,
    val,
    test
):
    print(
        "\n========== IRIS =========="
    )

    print(
        f"Train: {len(train)}"
    )

    print(
        f"Val  : {len(val)}"
    )

    print(
        f"Test : {len(test)}"
    )


def print_fingerprint_summary(
    train,
    val,
    test
):
    print(
        "\n========== FINGERPRINT =========="
    )

    print(
        f"Train: {len(train)}"
    )

    print(
        f"Val  : {len(val)}"
    )

    print(
        f"Test : {len(test)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    project_root = Path(
        r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
    )

    iris_root = (
        project_root
        / "outputs"
        / "quality_check"
        / "normalized"
    )

    fingerprint_root = (
        project_root
        / "outputs"
        / "fingerprint"
        / "preprocessed"
    )

    iris_train, iris_val, iris_test = split_iris(
        iris_root
    )

    (
        fingerprint_train,
        fingerprint_val,
        fingerprint_test
    ) = split_fingerprint(
        fingerprint_root
    )

    print_iris_summary(
        iris_train,
        iris_val,
        iris_test
    )

    print_fingerprint_summary(
        fingerprint_train,
        fingerprint_val,
        fingerprint_test
    )

    print(
        "\n========== EXPECTED =========="
    )

    print("Iris:")
    print("  270 train | 90 val | 90 test")

    print("Fingerprint:")
    print("  1350 train | 450 val | 450 test")

    print(
        "\nStructured split verification complete."
    )


if __name__ == "__main__":
    main()