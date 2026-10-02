from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

STRUCTURED_SPLIT = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "structured_split.json"
)


# ============================================================
# VERIFY IRIS SPLIT
# ============================================================

def verify_iris_split(split_data):
    print(
        "\n========== IRIS SPLIT =========="
    )

    required_splits = [
        "train",
        "val",
        "test"
    ]

    for subject in range(1, 46):
        subject_files = {
            "train": [],
            "val": [],
            "test": []
        }

        for split_name in required_splits:
            for file in split_data["iris"][split_name]:
                filename = Path(file).name

                if filename.startswith(
                    f"{subject}_"
                ):
                    subject_files[
                        split_name
                    ].append(filename)

        left_train = [
            file
            for file in subject_files["train"]
            if "_left_" in file.lower()
        ]

        right_train = [
            file
            for file in subject_files["train"]
            if "_right_" in file.lower()
        ]

        left_val = [
            file
            for file in subject_files["val"]
            if "_left_" in file.lower()
        ]

        right_val = [
            file
            for file in subject_files["val"]
            if "_right_" in file.lower()
        ]

        left_test = [
            file
            for file in subject_files["test"]
            if "_left_" in file.lower()
        ]

        right_test = [
            file
            for file in subject_files["test"]
            if "_right_" in file.lower()
        ]

        if (
            len(left_train) != 3
            or len(right_train) != 3
            or len(left_val) != 1
            or len(right_val) != 1
            or len(left_test) != 1
            or len(right_test) != 1
        ):
            raise ValueError(
                f"Invalid iris split for "
                f"subject {subject}: "
                f"left train={len(left_train)}, "
                f"right train={len(right_train)}, "
                f"left val={len(left_val)}, "
                f"right val={len(right_val)}, "
                f"left test={len(left_test)}, "
                f"right test={len(right_test)}"
            )

    print(
        "All 45 subjects verified."
    )

    print(
        "Each subject:"
    )

    print(
        "  Left eye  : 3 train | 1 val | 1 test"
    )

    print(
        "  Right eye : 3 train | 1 val | 1 test"
    )


# ============================================================
# VERIFY FINGERPRINT SPLIT
# ============================================================

def verify_fingerprint_split(split_data):
    print(
        "\n========== FINGERPRINT SPLIT =========="
    )

    required_splits = [
        "train",
        "val",
        "test"
    ]

    for subject in range(1, 46):
        subject_counts = {
            "train": 0,
            "val": 0,
            "test": 0
        }

        for split_name in required_splits:
            for file in (
                split_data["fingerprint"][split_name]
            ):
                file_path = Path(file)

                filename_subject = file_path.parent.name

                if filename_subject == str(subject):
                    subject_counts[
                        split_name
                    ] += 1

        if (
            subject_counts["train"] != 6
            or subject_counts["val"] != 2
            or subject_counts["test"] != 2
        ):
            raise ValueError(
                f"Invalid fingerprint split "
                f"for subject {subject}: "
                f"{subject_counts}"
            )

    print(
        "All 45 subjects verified."
    )

    print(
        "Each subject:"
    )

    print(
        "  6 train | 2 val | 2 test"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not STRUCTURED_SPLIT.exists():
        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{STRUCTURED_SPLIT}"
        )

    print(
        "Loading structured split..."
    )

    with open(
        STRUCTURED_SPLIT,
        "r",
        encoding="utf-8"
    ) as file:
        split_data = json.load(file)

    verify_iris_split(
        split_data
    )

    verify_fingerprint_split(
        split_data
    )

    print(
        "\n========== VERIFICATION COMPLETE =========="
    )

    print(
        "The structured split is valid."
    )


if __name__ == "__main__":
    main()