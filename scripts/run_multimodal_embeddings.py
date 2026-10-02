from pathlib import Path
import sys

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

from experiments.iris.embedding_model import (
    IrisEmbeddingModel
)


# ============================================================
# PATHS
# ============================================================

IRIS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "quality_check"
    / "normalized"
)

FINGERPRINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fingerprint"
    / "preprocessed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
)

IRIS_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "iris_embedding_best.pth"
)

FINGERPRINT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "fingerprint_classification_best.pth"
)

IRIS_OUTPUT = (
    OUTPUT_DIR
    / "iris_embeddings.pt"
)

FINGERPRINT_OUTPUT = (
    OUTPUT_DIR
    / "fingerprint_embeddings.pt"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DIM = 128


# ============================================================
# FINGERPRINT CLASSIFICATION MODEL
# ============================================================

class FingerprintClassificationModel(
    nn.Module
):

    def __init__(
        self,
        embedding_dim=128,
        num_classes=45
    ):

        super().__init__()

        self.backbone = resnet18(
            weights=None
        )

        old_conv = self.backbone.conv1

        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        self.backbone.conv1 = new_conv

        self.backbone.fc = nn.Identity()

        num_features = 512

        self.embedding = nn.Linear(
            num_features,
            embedding_dim
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        features = self.backbone(
            x
        )

        embedding = self.embedding(
            features
        )

        logits = self.classifier(
            embedding
        )

        return logits, embedding


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(file):

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

    image = image.unsqueeze(0)

    return image


# ============================================================
# EXTRACT IRIS SUBJECT
# ============================================================

def get_iris_subject(file):

    return int(
        file.name.split("_")[0]
    )


# ============================================================
# EXTRACT FINGERPRINT SUBJECT
# ============================================================

def get_fingerprint_subject(file):

    return int(
        file.parent.name
    )


# ============================================================
# LOAD IRIS MODEL
# ============================================================

def load_iris_model(device):

    model = IrisEmbeddingModel(
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    checkpoint = torch.load(
        IRIS_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# LOAD FINGERPRINT MODEL
# ============================================================

def load_fingerprint_model(device):

    model = FingerprintClassificationModel(
        embedding_dim=EMBEDDING_DIM,
        num_classes=45
    ).to(device)

    checkpoint = torch.load(
        FINGERPRINT_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# EXTRACT IRIS EMBEDDINGS
# ============================================================

def extract_iris_embeddings(
model,
files,
device
):

    embeddings = []
    filenames = []
    subjects = []

    with torch.no_grad():

        for index, file in enumerate(files):

            image = load_image(
                file
            ).to(device)

            embedding = model(
                image
            )

            embedding = F.normalize(
                embedding,
                p=2,
                dim=1
            )

            embeddings.append(
                embedding.squeeze(0).cpu()
            )

            filenames.append(
                file.name
            )

            subjects.append(
                get_iris_subject(file)
            )

            if (
                (index + 1) % 50
                == 0
            ):

                print(
                    f"  Processed "
                    f"{index + 1}/{len(files)}"
                )

    return (
        torch.stack(embeddings),
        filenames,
        subjects
    )




# ============================================================
# EXTRACT FINGERPRINT EMBEDDINGS
# ============================================================

def extract_fingerprint_embeddings(
    model,
    files,
    device
):

    embeddings = []
    filenames = []
    subjects = []

    with torch.no_grad():

        for index, file in enumerate(files):

            image = load_image(
                file
            ).to(device)

            _, embedding = model(
                image
            )

            embedding = F.normalize(
                embedding,
                p=2,
                dim=1
            )

            embeddings.append(
                embedding.squeeze(0).cpu()
            )

            filenames.append(
                file.name
            )

            subjects.append(
                get_fingerprint_subject(file)
            )

            if (
                (index + 1) % 50
                == 0
            ):

                print(
                    f"  Processed "
                    f"{index + 1}/{len(files)}"
                )

    return (
        torch.stack(embeddings),
        filenames,
        subjects
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        "\nLoading models..."
    )

    iris_model = load_iris_model(
        device
    )

    fingerprint_model = (
        load_fingerprint_model(
            device
        )
    )

    print(
        "Models loaded successfully."
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    iris_files = sorted(
        IRIS_DIR.glob("*.png")
    )

    fingerprint_files = sorted(
        FINGERPRINT_DIR.rglob("*.png")
    )

    print(
        f"\nIris files found: "
        f"{len(iris_files)}"
    )

    print(
        f"Fingerprint files found: "
        f"{len(fingerprint_files)}"
    )

    # --------------------------------------------------------
    # Iris embeddings
    # --------------------------------------------------------

    print(
        "\nExtracting iris embeddings..."
    )

    (
        iris_embeddings,
        iris_filenames,
        iris_subjects
    ) = extract_iris_embeddings(
        iris_model,
        iris_files,
        device
    )

    # --------------------------------------------------------
    # Fingerprint embeddings
    # --------------------------------------------------------

    print(
        "\nExtracting fingerprint embeddings..."
    )

    (
        fingerprint_embeddings,
        fingerprint_filenames,
        fingerprint_subjects
    ) = extract_fingerprint_embeddings(
        fingerprint_model,
        fingerprint_files,
        device
    )

    # --------------------------------------------------------
    # Save iris embeddings
    # --------------------------------------------------------

    torch.save(
        {
            "embeddings":
                iris_embeddings,

            "filenames":
                iris_filenames,

            "subjects":
                torch.tensor(iris_subjects)
        },
        IRIS_OUTPUT
    )

    # --------------------------------------------------------
    # Save fingerprint embeddings
    # --------------------------------------------------------

    torch.save(
        {
            "embeddings":
                fingerprint_embeddings,

            "filenames":
                fingerprint_filenames,

            "subjects":
                torch.tensor(fingerprint_subjects)
        },
        FINGERPRINT_OUTPUT
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n========== SUMMARY =========="
    )

    print(
        f"Iris embeddings: "
        f"{iris_embeddings.shape}"
    )

    print(
        f"Fingerprint embeddings: "
        f"{fingerprint_embeddings.shape}"
    )

    print(
        f"Iris subjects: "
        f"{len(set(iris_subjects))}"
    )

    print(
        f"Fingerprint subjects: "
        f"{len(set(fingerprint_subjects))}"
    )

    print(
        "\nSaved:"
    )

    print(
        IRIS_OUTPUT
    )

    print(
        FINGERPRINT_OUTPUT
    )


if __name__ == "__main__":
    main()

