"""Siamese metric-learning package for DVD cover recognition."""

from .dataset import (
    DVDReferencePairDataset,
    RandomDiscreteRotation,
    build_baseline_transform,
    build_inference_transform,
)
from .loss import ContrastiveLoss
from .model import BaselineSiameseNetwork
from .retrieval import (
    build_reference_gallery,
    encode_image,
    search_gallery,
)

__all__ = [
    "BaselineSiameseNetwork",
    "ContrastiveLoss",
    "DVDReferencePairDataset",
    "RandomDiscreteRotation",
    "build_baseline_transform",
    "build_inference_transform",
    "build_reference_gallery",
    "encode_image",
    "search_gallery",
]