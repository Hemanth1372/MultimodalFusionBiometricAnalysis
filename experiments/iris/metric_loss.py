import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):

    def __init__(self, margin=0.3):

        super().__init__()

        self.margin = margin

    def forward(self, embeddings, labels):

        distances = 1 - torch.mm(
            embeddings,
            embeddings.t()
        )

        losses = []

        for i in range(len(labels)):

            positive_mask = (
                labels == labels[i]
            )

            negative_mask = (
                labels != labels[i]
            )

            positive_mask[i] = False

            positives = distances[
                i
            ][positive_mask]

            negatives = distances[
                i
            ][negative_mask]

            if len(positives) == 0:
                continue

            hardest_positive = positives.max()
            hardest_negative = negatives.min()

            loss = F.relu(
                hardest_positive
                - hardest_negative
                + self.margin
            )

            losses.append(loss)

        if not losses:
            return torch.tensor(
                0.0,
                device=embeddings.device,
                requires_grad=True
            )

        return torch.stack(losses).mean()