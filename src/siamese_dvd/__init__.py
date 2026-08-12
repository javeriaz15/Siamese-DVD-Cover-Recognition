"""Siamese metric-learning package for DVD cover recognition."""

from .loss import ContrastiveLoss
from .model import BaselineSiameseNetwork

__all__ = [
    "BaselineSiameseNetwork",
    "ContrastiveLoss",
]