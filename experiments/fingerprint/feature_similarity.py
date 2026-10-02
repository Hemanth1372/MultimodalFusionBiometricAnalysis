from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fingerprint.config import PREPROCESSED_DIR
from src.fingerprint.dataset import split_fingerprint_dataset, FingerprintDataset
from src.fingerprint.model import FingerprintResNet18


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def extract_features(model, loader):

    features = []
    labels = []

    model.eval()

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(DEVICE)

            x = model.model.conv1(images)
            x = model.model.bn1(x)
            x = model.model.relu(x)
            x = model.model.maxpool(x)

            x = model.model.layer1(x)
            x = model.model.layer2(x)
            x = model.model.layer3(x)
            x = model.model.layer4(x)

            x = model.model.avgpool(x)
            x = torch.flatten(x, 1)

            x = F.normalize(
                x,
                p=2,
                dim=1
            )

            features.append(
                x.cpu()
            )

            labels.append(
                batch_labels
            )

    return (
        torch.cat(features),
        torch.cat(labels)
    )


def main():

    _, _, test_files = split_fingerprint_dataset(
        PREPROCESSED_DIR
    )

    subjects = sorted({
        int(file.parent.name)
        for file in test_files
    })

    subject_to_label = {
        subject: index
        for index, subject in enumerate(subjects)
    }

    dataset = FingerprintDataset(
        test_files,
        subject_to_label,
        training=False
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False
    )

    model = FingerprintResNet18(
        num_classes=45
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            PROJECT_ROOT
            / "models"
            / "fingerprint_resnet18_best.pth",
            map_location=DEVICE
        )
    )

    features, labels = extract_features(
        model,
        loader
    )

    similarity = features @ features.T

    correct = 0
    total = 0

    for i in range(len(features)):

        scores = similarity[i].clone()

        scores[i] = -1

        nearest = scores.argmax().item()

        if labels[nearest] == labels[i]:
            correct += 1

        total += 1

    accuracy = correct / total

    print(
        f"Nearest-neighbor accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Correct: {correct}/{total}"
    )

if __name__ == "__main__":
    main()