from pathlib import Path
import json

import torch
import torch.nn.functional as F


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

STRUCTURED_SPLIT = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "structured_split.json"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(path):

    data = torch.load(
        path,
        weights_only=False
    )

    embeddings = F.normalize(
        data["embeddings"],
        p=2,
        dim=1
    )

    return {
        "embeddings": embeddings,
        "subjects": data["subjects"],
        "filenames": data["filenames"],
    }


# ============================================================
# LOAD SAVED STRUCTURED SPLIT
# ============================================================

def load_structured_split(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CREATE INDICES FROM SAVED FILENAMES
# ============================================================

def create_split_indices(
    data,
    split_data,
    modality
):

    filename_to_index = {}

    for index, filename in enumerate(
        data["filenames"]
    ):

        filename_to_index[
            Path(filename).name
        ] = index

    train_indices = []
    val_indices = []
    test_indices = []

    for split_name, target in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices)
    ]:

        files = split_data[
            modality
        ][split_name]

        for file in files:

            filename = Path(file).name

            if filename not in filename_to_index:

                raise ValueError(
                    f"File not found in "
                    f"{modality} embeddings: "
                    f"{filename}"
                )

            target.append(
                filename_to_index[filename]
            )

    return (
        train_indices,
        val_indices,
        test_indices
    )


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(
    data,
    train_indices,
    val_indices,
    test_indices,
    modality
):

    all_indices = (
        train_indices
        + val_indices
        + test_indices
    )

    if len(all_indices) != len(
        data["filenames"]
    ):

        raise ValueError(
            f"{modality} split contains "
            f"{len(all_indices)} samples, "
            f"but embeddings contain "
            f"{len(data['filenames'])}."
        )

    if len(set(all_indices)) != len(
        all_indices
    ):

        raise ValueError(
            f"{modality} split contains "
            f"duplicate samples."
        )

    print(
        f"{modality} split verified: "
        f"{len(train_indices)} train | "
        f"{len(val_indices)} val | "
        f"{len(test_indices)} test"
    )


# ============================================================
# GET EYE TYPE FROM FILENAME
# ============================================================

def get_eye_type(filename):

    name = Path(filename).stem.lower()

    if "_left_" in name:

        return "left"

    if "_right_" in name:

        return "right"

    raise ValueError(
        f"Cannot determine eye type "
        f"from filename: {filename}"
    )


# ============================================================
# BUILD SEPARATE IRIS TEMPLATES
# ============================================================

def build_iris_templates(
    embeddings,
    subjects,
    filenames,
    indices
):

    subject_embeddings = {}

    for index in indices:

        subject = int(
            subjects[index].item()
        )

        eye = get_eye_type(
            filenames[index]
        )

        key = (
            subject,
            eye
        )

        subject_embeddings.setdefault(
            key,
            []
        ).append(
            embeddings[index]
        )

    templates = {}

    for key, values in subject_embeddings.items():

        template = torch.stack(
            values
        ).mean(
            dim=0
        )

        template = F.normalize(
            template.unsqueeze(0),
            p=2,
            dim=1
        ).squeeze(0)

        templates[key] = template

    return templates


# ============================================================
# BUILD FINGERPRINT SAMPLE SETS
# ============================================================

def build_fingerprint_sets(
    embeddings,
    subjects,
    indices
):

    subject_embeddings = {}

    for index in indices:

        subject = int(
            subjects[index].item()
        )

        subject_embeddings.setdefault(
            subject,
            []
        ).append(
            embeddings[index]
        )

    return subject_embeddings


# ============================================================
# CALCULATE IRIS SIMILARITY SCORES
# ============================================================

def get_iris_scores(
    query_embedding,
    templates
):

    subjects = sorted(
        set(
            subject
            for subject, eye in templates.keys()
        )
    )

    scores = {}

    for subject in subjects:

        left_template = templates.get(
            (subject, "left")
        )

        right_template = templates.get(
            (subject, "right")
        )

        candidate_scores = []

        if left_template is not None:

            candidate_scores.append(
                torch.dot(
                    query_embedding,
                    left_template
                ).item()
            )

        if right_template is not None:

            candidate_scores.append(
                torch.dot(
                    query_embedding,
                    right_template
                ).item()
            )

        if not candidate_scores:

            raise ValueError(
                f"No iris template found "
                f"for subject {subject}"
            )

        scores[subject] = max(
            candidate_scores
        )

    return scores


# ============================================================
# CALCULATE FINGERPRINT SIMILARITY SCORES
# ============================================================

def get_fingerprint_scores(
    query_embedding,
    fingerprint_sets
):

    subjects = sorted(
        fingerprint_sets.keys()
    )

    scores = {}

    for subject in subjects:

        subject_embeddings = torch.stack(
            fingerprint_sets[subject]
        )

        similarities = (
            subject_embeddings
            @ query_embedding
        )

        scores[subject] = similarities.max().item()

    return scores


# ============================================================
# Z-SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    subjects = sorted(
        scores.keys()
    )

    values = torch.tensor(
        [
            scores[subject]
            for subject in subjects
        ],
        dtype=torch.float32
    )

    mean = values.mean()

    std = values.std(
        unbiased=False
    )

    if std < 1e-8:

        normalized_values = (
            values - mean
        )

    else:

        normalized_values = (
            values - mean
        ) / std

    return {
        subject: normalized_values[index].item()
        for index, subject in enumerate(subjects)
    }


# ============================================================
# SINGLE IRIS MODALITY EVALUATION
# ============================================================

def evaluate_iris(
    embeddings,
    subjects,
    query_indices,
    templates
):

    correct = 0
    total = 0

    for index in query_indices:

        query_embedding = embeddings[index]

        true_subject = int(
            subjects[index].item()
        )

        scores = get_iris_scores(
            query_embedding,
            templates
        )

        predicted_subject = max(
            scores,
            key=scores.get
        )

        if predicted_subject == true_subject:

            correct += 1

        total += 1

    return (
        correct / total
        if total
        else 0.0
    )


# ============================================================
# SINGLE FINGERPRINT MODALITY EVALUATION
# ============================================================

def evaluate_fingerprint(
    embeddings,
    subjects,
    query_indices,
    fingerprint_sets
):

    correct = 0
    total = 0

    for index in query_indices:

        query_embedding = embeddings[index]

        true_subject = int(
            subjects[index].item()
        )

        scores = get_fingerprint_scores(
            query_embedding,
            fingerprint_sets
        )

        predicted_subject = max(
            scores,
            key=scores.get
        )

        if predicted_subject == true_subject:

            correct += 1

        total += 1

    return (
        correct / total
        if total
        else 0.0
    )


# ============================================================
# Z-SCORE NORMALIZED MULTIMODAL FUSION
# ============================================================

def evaluate_fusion(
    iris_embeddings,
    iris_subjects,
    iris_query_indices,
    iris_templates,
    fingerprint_embeddings,
    fingerprint_subjects,
    fingerprint_query_indices,
    fingerprint_sets,
    alpha
):

    correct = 0
    total = 0

    fingerprint_indices_by_subject = {}

    for index in fingerprint_query_indices:

        subject = int(
            fingerprint_subjects[index].item()
        )

        fingerprint_indices_by_subject.setdefault(
            subject,
            []
        ).append(index)

    subjects = sorted(
        fingerprint_sets.keys()
    )

    for iris_index in iris_query_indices:

        iris_subject = int(
            iris_subjects[iris_index].item()
        )

        iris_scores = get_iris_scores(
            iris_embeddings[iris_index],
            iris_templates
        )

        iris_scores = normalize_scores(
            iris_scores
        )

        matching_fingerprint_indices = (
            fingerprint_indices_by_subject.get(
                iris_subject,
                []
            )
        )

        for fingerprint_index in (
            matching_fingerprint_indices
        ):

            fingerprint_scores = (
                get_fingerprint_scores(
                    fingerprint_embeddings[
                        fingerprint_index
                    ],
                    fingerprint_sets
                )
            )

            fingerprint_scores = normalize_scores(
                fingerprint_scores
            )

            fusion_scores = {
                subject: (
                    alpha
                    * iris_scores[subject]
                    + (
                        1.0 - alpha
                    )
                    * fingerprint_scores[subject]
                )
                for subject in subjects
            }

            predicted_subject = max(
                fusion_scores,
                key=fusion_scores.get
            )

            if predicted_subject == iris_subject:

                correct += 1

            total += 1

    return (
        correct / total
        if total
        else 0.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading embeddings..."
    )

    iris = load_embeddings(
        IRIS_EMBEDDINGS
    )

    fingerprint = load_embeddings(
        FINGERPRINT_EMBEDDINGS
    )

    print(
        "Embeddings loaded."
    )

    print(
        f"Iris embeddings: "
        f"{iris['embeddings'].shape}"
    )

    print(
        f"Fingerprint embeddings: "
        f"{fingerprint['embeddings'].shape}"
    )

    # --------------------------------------------------------
    # Verify subject IDs
    # --------------------------------------------------------

    iris_subjects = set(
        iris["subjects"].tolist()
    )

    fingerprint_subjects = set(
        fingerprint["subjects"].tolist()
    )

    if iris_subjects != fingerprint_subjects:

        raise ValueError(
            "Iris and fingerprint "
            "subject sets differ."
        )

    print(
        f"Subjects: "
        f"{len(iris_subjects)}"
    )

    # --------------------------------------------------------
    # Load fixed structured split
    # --------------------------------------------------------

    print(
        "\nLoading structured split..."
    )

    split_data = load_structured_split(
        STRUCTURED_SPLIT
    )

    (
        iris_train,
        iris_val,
        iris_test
    ) = create_split_indices(
        iris,
        split_data,
        "iris"
    )

    (
        fingerprint_train,
        fingerprint_val,
        fingerprint_test
    ) = create_split_indices(
        fingerprint,
        split_data,
        "fingerprint"
    )

    # --------------------------------------------------------
    # Verify splits
    # --------------------------------------------------------

    print(
        "\n========== SPLIT =========="
    )

    verify_split(
        iris,
        iris_train,
        iris_val,
        iris_test,
        "Iris"
    )

    verify_split(
        fingerprint,
        fingerprint_train,
        fingerprint_val,
        fingerprint_test,
        "Fingerprint"
    )

    # --------------------------------------------------------
    # Build enrollment representations
    # --------------------------------------------------------

    iris_templates = build_iris_templates(
        iris["embeddings"],
        iris["subjects"],
        iris["filenames"],
        iris_train
    )

    fingerprint_sets = build_fingerprint_sets(
        fingerprint["embeddings"],
        fingerprint["subjects"],
        fingerprint_train
    )

    print(
        f"\nIris enrollment templates: "
        f"{len(iris_templates)}"
    )

    print(
        "Expected iris templates: "
        "90 "
        "(45 subjects x 2 eyes)"
    )

    print(
        f"Fingerprint enrollment sets: "
        f"{len(fingerprint_sets)}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n========== Z-SCORE NORMALIZED FUSION =========="
    )

    print(
        "Fingerprint aggregation: "
        "maximum similarity across enrolled fingers"
    )

    print(
        "Alpha = Iris weight"
    )

    print(
        "Fingerprint weight = 1 - Alpha"
    )

    print(
        "\nAlpha    Validation Accuracy"
    )

    print(
        "----------------------------"
    )

    validation_results = {}

    alpha_values = [
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
    ]

    for alpha in alpha_values:

        accuracy = evaluate_fusion(
            iris["embeddings"],
            iris["subjects"],
            iris_val,
            iris_templates,
            fingerprint["embeddings"],
            fingerprint["subjects"],
            fingerprint_val,
            fingerprint_sets,
            alpha
        )

        validation_results[
            alpha
        ] = accuracy

        print(
            f"{alpha:.1f}      "
            f"{accuracy:.4f}"
        )

    # --------------------------------------------------------
    # Select alpha using validation
    # --------------------------------------------------------

    best_alpha = max(
        validation_results,
        key=validation_results.get
    )

    best_val_accuracy = (
        validation_results[
            best_alpha
        ]
    )

    print(
        "\nBest Alpha from Validation: "
        f"{best_alpha:.1f}"
    )

    print(
        "Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\n========== FINAL TEST =========="
    )

    iris_accuracy = evaluate_iris(
        iris["embeddings"],
        iris["subjects"],
        iris_test,
        iris_templates
    )

    fingerprint_accuracy = evaluate_fingerprint(
        fingerprint["embeddings"],
        fingerprint["subjects"],
        fingerprint_test,
        fingerprint_sets
    )

    fusion_accuracy = evaluate_fusion(
        iris["embeddings"],
        iris["subjects"],
        iris_test,
        iris_templates,
        fingerprint["embeddings"],
        fingerprint["subjects"],
        fingerprint_test,
        fingerprint_sets,
        best_alpha
    )

    print(
        f"Iris-only Test Accuracy: "
        f"{iris_accuracy:.4f}"
    )

    print(
        f"Fingerprint-only Test Accuracy: "
        f"{fingerprint_accuracy:.4f}"
    )

    print(
        f"Selected Alpha: "
        f"{best_alpha:.1f}"
    )

    print(
        f"Normalized Fusion Test Accuracy: "
        f"{fusion_accuracy:.4f}"
    )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()

