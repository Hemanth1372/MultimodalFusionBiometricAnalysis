from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from .model import IrisResNet18
from .split import split_iris_dataset


PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "quality_check"
    / "normalized"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "iris_resnet18_best.pth"
)


def load_image(file):

    image = cv2.imread(
        str(file),
        cv2.IMREAD_GRAYSCALE
    )

    image = (
        torch.from_numpy(image)
        .float()
        / 255.0
    )

    return image.unsqueeze(0).unsqueeze(0)


def main():

    device = torch.device("cpu")

    _, _, test_files = split_iris_dataset(
        NORMALIZED_DIR
    )

    model = IrisResNet18(
        num_classes=45
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    features = []

    def hook(module, inputs, output):

        features.append(
            output.detach()
        )

    handle = model.model.avgpool.register_forward_hook(
        hook
    )

    with torch.no_grad():

        for file in test_files:

            image = load_image(file).to(device)

            model(image)

    handle.remove()

    features = torch.cat(
        features
    )

    features = features.flatten(
        start_dim=1
    )

    features = F.normalize(
        features,
        p=2,
        dim=1
    )

    print(
        f"Test images: {len(test_files)}"
    )

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Feature norm range: "
        f"{features.norm(dim=1).min():.4f} - "
        f"{features.norm(dim=1).max():.4f}"
    )


if __name__ == "__main__":
    main()