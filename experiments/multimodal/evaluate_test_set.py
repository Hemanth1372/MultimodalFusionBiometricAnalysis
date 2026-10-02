import json
from pathlib import Path
import sys
import torch

# ============================================================
# PROJECT PATH
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics")
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
    PROJECT_ROOT / "outputs" / "multimodal" / "structured_split.json"
)

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cpu")


# ============================================================
# LOAD TEST SPLIT
# ============================================================
def load_test_split():
    with open(STRUCTURED_SPLIT, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    iris_test = split_data["iris"]["test"]
    fingerprint_test = split_data["fingerprint"]["test"]

    return iris_test, fingerprint_test


# ============================================================
# GROUP TEST FILES BY SUBJECT
# ============================================================
def group_iris_by_subject(paths):
    subjects = {}

    for path in paths:
        path = Path(path)
        subject = int(path.name.split("_")[0])
        subjects.setdefault(subject, []).append(path)

    return subjects


def group_fingerprint_by_subject(paths):
    subjects = {}

    for path in paths:
        path = Path(path)
        subject = int(path.name.split("__")[0])
        subjects.setdefault(subject, []).append(path)

    return subjects


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load test split
    # --------------------------------------------------------
    print("\nLoading test split...")
    iris_test_paths, fingerprint_test_paths = load_test_split()

    iris_test = group_iris_by_subject(iris_test_paths)
    fingerprint_test = group_fingerprint_by_subject(fingerprint_test_paths)

    print(f"Iris test samples        : {len(iris_test_paths)}")
    print(f"Fingerprint test samples : {len(fingerprint_test_paths)}")
    print(f"Test subjects            : {len(iris_test)}")

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
        fingerprint_enrollment,
    ) = load_enrollment_data()

    print(f"Iris enrollment subjects: {len(iris_enrollment)}")
    print(
        f"Fingerprint enrollment subjects: {len(fingerprint_enrollment)}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    iris_correct = 0
    fingerprint_correct = 0
    fusion_correct = 0

    # Fusion diagnostics
    iris_wrong_fusion_correct = 0
    iris_correct_fusion_wrong = 0
    both_wrong = 0

    total = 0
    alpha = 0.7

    print("\n========== TEST SET EVALUATION ==========")

    for subject in sorted(iris_test.keys()):
        iris_samples = iris_test[subject]
        fingerprint_samples = fingerprint_test[subject]

        if len(iris_samples) != len(fingerprint_samples):
            raise ValueError(
                f"Test sample mismatch for Subject {subject}: "
                f"{len(iris_samples)} iris vs "
                f"{len(fingerprint_samples)} fingerprint"
            )

        for iris_path, fingerprint_path in zip(
            iris_samples, fingerprint_samples
        ):
            iris_embedding = extract_iris_embedding(
                iris_model, iris_path
            )

            fingerprint_embedding = extract_fingerprint_embedding(
                fingerprint_model, fingerprint_path
            )

            # ------------------------------------------------
            # Iris prediction
            # ------------------------------------------------
            iris_scores = get_iris_scores(
                iris_embedding, iris_enrollment
            )
            iris_prediction = max(iris_scores, key=iris_scores.get)

            # ------------------------------------------------
            # Fingerprint prediction
            # ------------------------------------------------
            fingerprint_scores = get_fingerprint_scores(
                fingerprint_embedding, fingerprint_enrollment
            )
            fingerprint_prediction = max(
                fingerprint_scores, key=fingerprint_scores.get
            )

            # ------------------------------------------------
            # Fusion
            # ------------------------------------------------
            fused_scores, _, _ = fuse_scores(
                iris_scores, fingerprint_scores, alpha=alpha
            )
            fusion_prediction = max(
                fused_scores, key=fused_scores.get
            )

            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------
            iris_is_correct = (
                iris_prediction == subject
            )

            fusion_is_correct = (
                fusion_prediction == subject
            )

            if iris_is_correct:
                iris_correct += 1

            if fingerprint_prediction == subject:
                fingerprint_correct += 1

            if fusion_is_correct:
                fusion_correct += 1

            # ------------------------------------------------
            # Fusion Diagnostics
            # ------------------------------------------------
            if not iris_is_correct and fusion_is_correct:
                iris_wrong_fusion_correct += 1

            elif iris_is_correct and not fusion_is_correct:
                iris_correct_fusion_wrong += 1

            elif not iris_is_correct and not fusion_is_correct:
                both_wrong += 1

            total += 1

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------
    iris_accuracy = iris_correct / total
    fingerprint_accuracy = fingerprint_correct / total
    fusion_accuracy = fusion_correct / total

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Total test samples       : {total}")
    print(f"Iris correct              : {iris_correct}/{total}")
    print(f"Fingerprint correct       : {fingerprint_correct}/{total}")
    print(f"Fusion correct            : {fusion_correct}/{total}")
    print()
    print(f"Iris-only accuracy        : {iris_accuracy * 100:.2f}%")
    print(
        f"Fingerprint-only accuracy : {fingerprint_accuracy * 100:.2f}%"
    )
    print(
        f"Multimodal fusion accuracy: {fusion_accuracy * 100:.2f}%"
    )
    print()
    print("Fusion weights:")
    print(f"  Iris         : {alpha:.2f}")
    print(f"  Fingerprint : {1.0 - alpha:.2f}")

    # --------------------------------------------------------
    # Fusion Diagnostics
    # --------------------------------------------------------
    print("\n========== FUSION DIAGNOSTICS ==========")
    print(
        f"Iris wrong -> Fusion correct : "
        f"{iris_wrong_fusion_correct}"
    )
    print(
        f"Iris correct -> Fusion wrong : "
        f"{iris_correct_fusion_wrong}"
    )
    print(
        f"Both wrong                   : "
        f"{both_wrong}"
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

