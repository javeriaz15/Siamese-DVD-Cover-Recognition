"""Siamese metric-learning package for DVD cover recognition."""

from .dataset import (
    DVDReferencePairDataset,
    RandomDiscreteRotation,
    build_baseline_transform,
    build_inference_transform,
)
from .evaluation import DVDEvaluationDataset
from .loss import ContrastiveLoss
from .model import BaselineSiameseNetwork
from .preprocessing import (
    preprocess_image_file,
    rectify_dvd_cover,
)
from .retrieval import (
    build_reference_gallery,
    encode_image,
    search_gallery,
)

__all__ = [
    "BaselineSiameseNetwork",
    "ContrastiveLoss",
    "DVDReferencePairDataset",
    "DVDEvaluationDataset",
    "RandomDiscreteRotation",
    "build_baseline_transform",
    "build_inference_transform",
    "build_reference_gallery",
    "encode_image",
    "preprocess_image_file",
    "rectify_dvd_cover",
    "search_gallery",
]