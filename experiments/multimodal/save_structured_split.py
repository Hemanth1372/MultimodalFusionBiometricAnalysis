from pathlib import Path
import json

from structured_split import (
    split_iris,
    split_fingerprint
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

    output_dir = (
        project_root
        / "outputs"
        / "multimodal"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Creating structured splits...")

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

    split_data = {
        "iris": {
            "train": [
                str(file)
                for file in iris_train
            ],
            "val": [
                str(file)
                for file in iris_val
            ],
            "test": [
                str(file)
                for file in iris_test
            ]
        },
        "fingerprint": {
            "train": [
                str(file)
                for file in fingerprint_train
            ],
            "val": [
                str(file)
                for file in fingerprint_val
            ],
            "test": [
                str(file)
                for file in fingerprint_test
            ]
        }
    }

    output_file = (
        output_dir
        / "structured_split.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            split_data,
            file,
            indent=4
        )

    print(
        "\n========== SUMMARY =========="
    )

    print(
        f"Iris train       : "
        f"{len(iris_train)}"
    )

    print(
        f"Iris validation  : "
        f"{len(iris_val)}"
    )

    print(
        f"Iris test        : "
        f"{len(iris_test)}"
    )

    print(
        f"Fingerprint train      : "
        f"{len(fingerprint_train)}"
    )

    print(
        f"Fingerprint validation : "
        f"{len(fingerprint_val)}"
    )

    print(
        f"Fingerprint test       : "
        f"{len(fingerprint_test)}"
    )

    print(
        f"\nSaved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()