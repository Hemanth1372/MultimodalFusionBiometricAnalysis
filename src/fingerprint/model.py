import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FingerprintResNet18(nn.Module):

    def __init__(self, num_classes=45):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        old_conv = self.model.conv1

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

        self.model.conv1 = new_conv

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)
    