"""Siamese neural network architectures for DVD cover recognition."""

import torch
from torch import nn


class BaselineSiameseNetwork(nn.Module):
    """Original ECE 7650 Siamese CNN architecture.

    The network maps a grayscale 100x100 DVD-cover image to a compact
    embedding. Two images are passed independently through the same
    shared network and their embeddings can then be compared using
    Euclidean distance.

    This implementation preserves the architecture used in the original
    university project while exposing a reusable ``encode`` method for
    inference and retrieval.
    """

    def __init__(self, input_size: int = 100, embedding_dim: int = 5):
        super().__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        flattened_features = 8 * input_size * input_size

        self.embedding_head = nn.Sequential(
            nn.Linear(flattened_features, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images into embedding vectors."""
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        return self.embedding_head(features)

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings for a pair of input images."""
        output1 = self.encode(input1)
        output2 = self.encode(input2)

        return output1, output2