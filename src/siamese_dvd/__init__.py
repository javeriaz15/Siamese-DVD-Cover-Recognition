"""Siamese metric-learning package for DVD cover recognition."""

from .dataset import (
    DVDReferencePairDataset,
    RandomDiscreteRotation,
    build_baseline_transform,
)
from .loss import ContrastiveLoss
from .model import BaselineSiameseNetwork

__all__ = [
    "BaselineSiameseNetwork",
    "ContrastiveLoss",
    "DVDReferencePairDataset",
    "RandomDiscreteRotation",
    "build_baseline_transform",
]