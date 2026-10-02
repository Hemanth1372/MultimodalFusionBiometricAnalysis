import json
from pathlib import Path
import sys

import cv2
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

# ============================================================
# PROJECT PATH
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics")
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# PATHS
# ============================================================
IRIS_CHECKPOINT = PROJECT_ROOT / "models" / "iris_embedding_best.pth"
FINGERPRINT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "fingerprint_classification_best.pth"
)
STRUCTURED_SPLIT = (
    PROJECT_ROOT / "outputs" / "multimodal" / "structured_split.json"
)
IRIS_EMBEDDINGS = PROJECT_ROOT / "outputs" / "multimodal" / "iris_embeddings.pt"
FINGERPRINT_EMBEDDINGS = (
    PROJECT_ROOT / "outputs" / "multimodal" / "fingerprint_embeddings.pt"
)

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cpu")


# ============================================================
# IRIS MODEL
# ============================================================
def create_iris_model():
    from experiments.iris.embedding_model import IrisEmbeddingModel

    model = IrisEmbeddingModel(embedding_dim=128)

    checkpoint = torch.load(
        IRIS_CHECKPOINT, map_location=DEVICE, weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IRIS INFERENCE
# ============================================================
def extract_iris_embedding(model, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Failed to read iris image: {image_path}")

    image = torch.from_numpy(image).float() / 255.0

    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        embedding = model(image)

    return F.normalize(embedding, p=2, dim=1)


# ============================================================
# FINGERPRINT MODEL
# ============================================================
class FingerprintClassificationModel(torch.nn.Module):
    def __init__(self, embedding_dim=128, num_classes=45):
        super().__init__()

        self.backbone = resnet18(weights=None)
        old_conv = self.backbone.conv1

        self.backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        self.backbone.fc = torch.nn.Identity()
        self.embedding = torch.nn.Linear(512, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)

        return logits, embedding


def create_fingerprint_model():
    model = FingerprintClassificationModel(
        embedding_dim=128, num_classes=45
    )

    checkpoint = torch.load(
        FINGERPRINT_CHECKPOINT, map_location=DEVICE, weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# FINGERPRINT INFERENCE
# ============================================================
def extract_fingerprint_embedding(model, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Failed to read fingerprint image: {image_path}")

    image = cv2.resize(image, (224, 224))
    image = torch.from_numpy(image).float() / 255.0

    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        _, embedding = model(image)

    return F.normalize(embedding, p=2, dim=1)


# ============================================================
# PATH MATCHING HELPERS
# ============================================================
def normalize_path(path):
    return str(Path(path).resolve().as_posix().lower())


def build_path_lookup(paths):
    lookup = {}

    for path in paths:
        normalized = normalize_path(path)
        basename = Path(path).name.lower()

        lookup[normalized] = path
        lookup.setdefault(basename, path)

    return lookup


# ============================================================
# LOAD ENROLLMENT DATA
# ============================================================
def load_enrollment_data():
    with open(STRUCTURED_SPLIT, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    iris_data = torch.load(
        IRIS_EMBEDDINGS, map_location="cpu", weights_only=False
    )

    fingerprint_data = torch.load(
        FINGERPRINT_EMBEDDINGS, map_location="cpu", weights_only=False
    )

    iris_train_paths = split_data["iris"]["train"]
    fingerprint_train_paths = split_data["fingerprint"]["train"]

    iris_train_lookup = build_path_lookup(iris_train_paths)
    fingerprint_train_lookup = build_path_lookup(fingerprint_train_paths)

    iris_embeddings = {}

    for index, filename in enumerate(iris_data["filenames"]):
        path = Path(filename)
        normalized_path = normalize_path(path)
        basename = path.name.lower()

        if (
            normalized_path in iris_train_lookup
            or basename in iris_train_lookup
        ):
            subject = int(path.name.split("_")[0])

            embedding = F.normalize(
                iris_data["embeddings"][index], p=2, dim=0
            )

            iris_embeddings.setdefault(subject, []).append(embedding)

    fingerprint_embeddings = {}

    for index, filename in enumerate(fingerprint_data["filenames"]):
        path = Path(filename)
        normalized_path = normalize_path(path)
        basename = path.name.lower()

        if (
            normalized_path in fingerprint_train_lookup
            or basename in fingerprint_train_lookup
        ):
            subject = int(path.name.split("__")[0])

            embedding = F.normalize(
                fingerprint_data["embeddings"][index], p=2, dim=0
            )

            fingerprint_embeddings.setdefault(subject, []).append(
                embedding
            )

    if not iris_embeddings:
        raise ValueError(
            "No iris enrollment embeddings matched the structured split."
        )

    if not fingerprint_embeddings:
        raise ValueError(
            "No fingerprint enrollment embeddings matched the structured split."
        )

    return iris_embeddings, fingerprint_embeddings


# ============================================================
# IRIS SCORES
# ============================================================
def get_iris_scores(query_embedding, iris_enrollment):
    scores = {}
    query = query_embedding.squeeze(0)

    for subject, embeddings in iris_enrollment.items():
        enrolled = torch.stack(embeddings)
        similarities = torch.matmul(enrolled, query)
        scores[subject] = similarities.max().item()

    return scores


# ============================================================
# FINGERPRINT SCORES
# ============================================================
def get_fingerprint_scores(query_embedding, fingerprint_enrollment):
    scores = {}
    query = query_embedding.squeeze(0)

    for subject, embeddings in fingerprint_enrollment.items():
        enrolled = torch.stack(embeddings)
        similarities = torch.matmul(enrolled, query)
        scores[subject] = similarities.max().item()

    return scores


# ============================================================
# Z-SCORE NORMALIZATION
# ============================================================
def z_score_normalize(scores):
    values = torch.tensor(list(scores.values()), dtype=torch.float32)

    mean = values.mean()
    std = values.std()

    if std < 1e-8:
        normalized = torch.zeros_like(values)
    else:
        normalized = (values - mean) / std

    return {
        subject: normalized[index].item()
        for index, subject in enumerate(scores.keys())
    }


# ============================================================
# SCORE FUSION
# ============================================================
def fuse_scores(iris_scores, fingerprint_scores, alpha=0.6):
    iris_normalized = z_score_normalize(iris_scores)
    fingerprint_normalized = z_score_normalize(fingerprint_scores)

    subjects = sorted(iris_scores.keys())
    fused_scores = {}

    for subject in subjects:
        fused_scores[subject] = (
            alpha * iris_normalized[subject]
            + (1.0 - alpha) * fingerprint_normalized[subject]
        )

    return fused_scores, iris_normalized, fingerprint_normalized


# ============================================================
# PREDICTION
# ============================================================
def predict_subject(
    iris_embedding,
    fingerprint_embedding,
    iris_enrollment,
    fingerprint_enrollment,
    alpha=0.6,
):
    iris_scores = get_iris_scores(iris_embedding, iris_enrollment)
    fingerprint_scores = get_fingerprint_scores(
        fingerprint_embedding, fingerprint_enrollment
    )

    (
        fused_scores,
        iris_normalized,
        fingerprint_normalized,
    ) = fuse_scores(iris_scores, fingerprint_scores, alpha=alpha)

    predicted_subject = max(fused_scores, key=fused_scores.get)

    return (
        predicted_subject,
        iris_scores,
        fingerprint_scores,
        iris_normalized,
        fingerprint_normalized,
        fused_scores,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) != 3:
        print("\nUsage:")
        print(
            'python scripts\\multimodal_inference.py "<iris_image>" "<fingerprint_image>"'
        )
        print("\nExample:")
        print(
            "python scripts\\multimodal_inference.py "
            '"outputs\\quality_check\\normalized\\1_left_aeval1.png" '
            '"outputs\\fingerprint\\preprocessed\\1\\1__M_Left_index_finger.png"'
        )
        sys.exit(1)

    test_iris_image = Path(sys.argv[1])
    test_fingerprint_image = Path(sys.argv[2])

    if not test_iris_image.exists():
        raise FileNotFoundError(
            f"Iris image not found:\n{test_iris_image}"
        )

    if not test_fingerprint_image.exists():
        raise FileNotFoundError(
            f"Fingerprint image not found:\n{test_fingerprint_image}"
        )

    print(f"Device: {DEVICE}")
    print("\nInput images:")
    print(f"Iris        : {test_iris_image}")
    print(f"Fingerprint : {test_fingerprint_image}")

    # --------------------------------------------------------
    # Load iris model
    # --------------------------------------------------------
    print("\nLoading iris model...")
    iris_model = create_iris_model()
    print("Iris model loaded.")

    # --------------------------------------------------------
    # Extract iris embedding
    # --------------------------------------------------------
    print("\nExtracting iris embedding...")
    iris_embedding = extract_iris_embedding(iris_model, test_iris_image)
    print(f"Iris embedding shape: {iris_embedding.shape}")
    print(f"Iris embedding norm: {iris_embedding.norm().item():.4f}")

    # --------------------------------------------------------
    # Load fingerprint model
    # --------------------------------------------------------
    print("\nLoading fingerprint model...")
    fingerprint_model = create_fingerprint_model()
    print("Fingerprint model loaded.")

    # --------------------------------------------------------
    # Extract fingerprint embedding
    # --------------------------------------------------------
    print("\nExtracting fingerprint embedding...")
    fingerprint_embedding = extract_fingerprint_embedding(
        fingerprint_model, test_fingerprint_image
    )
    print(f"Fingerprint embedding shape: {fingerprint_embedding.shape}")
    print(
        f"Fingerprint embedding norm: {fingerprint_embedding.norm().item():.4f}"
    )

    # --------------------------------------------------------
    # Load enrollment data
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

    print("\n========== INFERENCE PIPELINE READY ==========")
    print("Iris embedding       : 128-D")
    print("Fingerprint embedding: 128-D")
    print(f"Enrollment subjects  : {len(iris_enrollment)}")

    # --------------------------------------------------------
    # Multimodal prediction
    # --------------------------------------------------------
    print("\n========== MULTIMODAL PREDICTION ==========")

    (
        predicted_subject,
        iris_scores,
        fingerprint_scores,
        iris_normalized,
        fingerprint_normalized,
        fused_scores,
    ) = predict_subject(
        iris_embedding,
        fingerprint_embedding,
        iris_enrollment,
        fingerprint_enrollment,
        alpha=0.6,
    )

    iris_prediction = max(iris_scores, key=iris_scores.get)
    fingerprint_prediction = max(
        fingerprint_scores, key=fingerprint_scores.get
    )

    print(f"Iris-only prediction        : Subject {iris_prediction}")
    print(
        f"Fingerprint-only prediction : Subject {fingerprint_prediction}"
    )
    print(f"Fused prediction            : Subject {predicted_subject}")

    print("\nFusion weights:")
    print("  Iris         : 0.60")
    print("  Fingerprint : 0.40")


if __name__ == "__main__":
    main()