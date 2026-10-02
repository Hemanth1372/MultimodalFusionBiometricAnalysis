from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

IRIS_EMBEDDINGS = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "iris_embeddings.pt"
)

FINGERPRINT_EMBEDDINGS = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "fingerprint_embeddings.pt"
)


# ============================================================
# SETTINGS
# ============================================================

SUBJECT_TO_INSPECT = 1


# ============================================================
# INSPECT IRIS EMBEDDINGS
# ============================================================

def inspect_iris():
    data = torch.load(
        IRIS_EMBEDDINGS,
        weights_only=False
    )

    print("\n========== IRIS ==========")

    files = [
        filename
        for filename, subject_id in zip(
            data["filenames"],
            data["subjects"].tolist()
        )
        if subject_id == SUBJECT_TO_INSPECT
    ]

    for index, filename in enumerate(files):
        print(
            f"{index}: {filename}"
        )


# ============================================================
# INSPECT FINGERPRINT EMBEDDINGS
# ============================================================

def inspect_fingerprint():
    data = torch.load(
        FINGERPRINT_EMBEDDINGS,
        weights_only=False
    )

    print("\n========== FINGERPRINT ==========")

    files = [
        filename
        for filename, subject_id in zip(
            data["filenames"],
            data["subjects"].tolist()
        )
        if subject_id == SUBJECT_TO_INSPECT
    ]

    for index, filename in enumerate(files):
        print(
            f"{index}: {filename}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    inspect_iris()
    inspect_fingerprint()


if __name__ == "__main__":
    main()