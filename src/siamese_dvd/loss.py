"""Loss functions for Siamese metric learning."""

import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning image similarity embeddings.

    Label convention:
        0 = similar pair
        1 = dissimilar pair

    Similar pairs are encouraged to have small Euclidean distance.
    Dissimilar pairs are encouraged to be separated by at least ``margin``.
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        distance = F.pairwise_distance(
            embedding1,
            embedding2,
            keepdim=True,
        )

        label = label.float().view_as(distance)

        loss = torch.mean(
            (1 - label) * distance.pow(2)
            + label * torch.clamp(
                self.margin - distance,
                min=0.0,
            ).pow(2)
        )

        return loss