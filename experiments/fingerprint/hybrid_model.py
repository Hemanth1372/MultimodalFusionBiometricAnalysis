import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FingerprintHybridModel(nn.Module):

    def __init__(
        self,
        num_classes=45,
        embedding_dim=128
    ):

        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
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

        with torch.no_grad():

            new_conv.weight[:] = (
                old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

        self.backbone.conv1 = new_conv

        num_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.embedding = nn.Linear(
            num_features,
            embedding_dim
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        features = self.backbone(x)

        embedding = self.embedding(
            features
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        logits = self.classifier(
            embedding
        )

        return embedding, logits
