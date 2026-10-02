import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class IrisEmbeddingModel(nn.Module):

    def __init__(self, embedding_dim=128):

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

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            num_features,
            embedding_dim
        )

    def forward(self, x):

        embedding = self.backbone(x)

        return nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )