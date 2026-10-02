import json
from pathlib import Path
import sys
import torch

# ============================================================
# PROJECT PATH
# ============================================================
PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORT INFERENCE FUNCTIONS
# ============================================================
from scripts.multimodal_inference import (
    create_fingerprint_model,
    create_iris_model,
    extract_fingerprint_embedding,
    extract_iris_embedding,
    fuse_scores,
    get_fingerprint_scores,
    get_iris_scores,
    load_enrollment_data,
)

# ============================================================
# PATHS
# ============================================================
STRUCTURED_SPLIT = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "structured_split.json"
)

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cpu")


# ============================================================
# LOAD VALIDATION SPLIT
# ============================================================
def load_validation_split():

    with open(
        STRUCTURED_SPLIT,
        "r",
        encoding="utf-8"
    ) as file:

        split_data = json.load(file)

    return (
        split_data["iris"]["val"],
        split_data["fingerprint"]["val"]
    )


# ============================================================
# GROUP FILES BY SUBJECT
# ============================================================
def group_iris_by_subject(paths):

    subjects = {}

    for path in paths:

        path = Path(path)

        subject = int(
            path.name.split("_")[0]
        )

        subjects.setdefault(
            subject,
            []
        ).append(path)

    return subjects


def group_fingerprint_by_subject(paths):

    subjects = {}

    for path in paths:

        path = Path(path)

        subject = int(
            path.name.split("__")[0]
        )

        subjects.setdefault(
            subject,
            []
        ).append(path)

    return subjects


# ============================================================
# MAIN
# ============================================================
def main():

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------
    print("\nLoading validation split...")

    (
        iris_paths,
        fingerprint_paths
    ) = load_validation_split()

    iris_validation = group_iris_by_subject(
        iris_paths
    )

    fingerprint_validation = (
        group_fingerprint_by_subject(
            fingerprint_paths
        )
    )

    print(
        f"Iris validation samples        : "
        f"{len(iris_paths)}"
    )

    print(
        f"Fingerprint validation samples : "
        f"{len(fingerprint_paths)}"
    )

    print(
        f"Validation subjects            : "
        f"{len(iris_validation)}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------
    print("\nLoading iris model...")

    iris_model = create_iris_model()

    print("Iris model loaded.")

    print("\nLoading fingerprint model...")

    fingerprint_model = create_fingerprint_model()

    print("Fingerprint model loaded.")

    # --------------------------------------------------------
    # Load enrollment
    # --------------------------------------------------------
    print("\nLoading enrollment data...")

    (
        iris_enrollment,
        fingerprint_enrollment
    ) = load_enrollment_data()

    print(
        f"Iris enrollment subjects: "
        f"{len(iris_enrollment)}"
    )

    print(
        f"Fingerprint enrollment subjects: "
        f"{len(fingerprint_enrollment)}"
    )

    # --------------------------------------------------------
    # Calculate scores ONCE
    # --------------------------------------------------------
    all_scores = []

    print(
        "\nGenerating validation scores..."
    )

    for subject in sorted(
        iris_validation.keys()
    ):

        iris_samples = iris_validation[
            subject
        ]

        fingerprint_samples = (
            fingerprint_validation[
                subject
            ]
        )

        if len(iris_samples) != len(
            fingerprint_samples
        ):

            raise ValueError(
                f"Validation sample mismatch "
                f"for Subject {subject}"
            )

        for (
            iris_path,
            fingerprint_path
        ) in zip(
            iris_samples,
            fingerprint_samples
        ):

            # ------------------------------------------------
            # Iris embedding
            # ------------------------------------------------
            iris_embedding = (
                extract_iris_embedding(
                    iris_model,
                    iris_path
                )
            )

            # ------------------------------------------------
            # Fingerprint embedding
            # ------------------------------------------------
            fingerprint_embedding = (
                extract_fingerprint_embedding(
                    fingerprint_model,
                    fingerprint_path
                )
            )

            # ------------------------------------------------
            # Individual modality scores
            # ------------------------------------------------
            iris_scores = get_iris_scores(
                iris_embedding,
                iris_enrollment
            )

            fingerprint_scores = (
                get_fingerprint_scores(
                    fingerprint_embedding,
                    fingerprint_enrollment
                )
            )

            all_scores.append(
                (
                    subject,
                    iris_scores,
                    fingerprint_scores
                )
            )

    print(
        f"Generated scores for "
        f"{len(all_scores)} samples."
    )

    # --------------------------------------------------------
    # Individual modality accuracy
    # --------------------------------------------------------
    iris_correct = 0
    fingerprint_correct = 0

    for (
        subject,
        iris_scores,
        fingerprint_scores
    ) in all_scores:

        iris_prediction = max(
            iris_scores,
            key=iris_scores.get
        )

        fingerprint_prediction = max(
            fingerprint_scores,
            key=fingerprint_scores.get
        )

        if iris_prediction == subject:
            iris_correct += 1

        if fingerprint_prediction == subject:
            fingerprint_correct += 1

    total = len(all_scores)

    # --------------------------------------------------------
    # Alpha sweep
    # --------------------------------------------------------
    print(
        "\n========== ALPHA SWEEP =========="
    )

    print(
        "Alpha | Iris Weight | Fingerprint Weight "
        "| Fusion Accuracy"
    )

    print(
        "------|-------------|-------------------"
        "|----------------"
    )

    results = []

    for alpha in [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ]:

        fusion_correct = 0

        for (
            subject,
            iris_scores,
            fingerprint_scores
        ) in all_scores:

            fused_scores, _, _ = fuse_scores(
                iris_scores,
                fingerprint_scores,
                alpha=alpha
            )

            prediction = max(
                fused_scores,
                key=fused_scores.get
            )

            if prediction == subject:
                fusion_correct += 1

        accuracy = (
            fusion_correct / total * 100
        )

        results.append(
            (
                alpha,
                fusion_correct,
                accuracy
            )
        )

        print(
            f"{alpha:.1f}   | "
            f"{alpha:.2f}        | "
            f"{1.0 - alpha:.2f}              | "
            f"{accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # Best validation alpha
    # --------------------------------------------------------
    best_alpha, best_correct, best_accuracy = max(
        results,
        key=lambda x: x[2]
    )

    print(
        "\n========== VALIDATION SUMMARY =========="
    )

    print(
        f"Iris-only accuracy        : "
        f"{iris_correct / total * 100:.2f}%"
    )

    print(
        f"Fingerprint-only accuracy : "
        f"{fingerprint_correct / total * 100:.2f}%"
    )

    print(
        f"Best validation alpha     : "
        f"{best_alpha:.1f}"
    )

    print(
        f"Best fusion accuracy      : "
        f"{best_accuracy:.2f}% "
        f"({best_correct}/{total})"
    )

    print(
        f"Iris weight               : "
        f"{best_alpha:.2f}"
    )

    print(
        f"Fingerprint weight        : "
        f"{1.0 - best_alpha:.2f}"
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

