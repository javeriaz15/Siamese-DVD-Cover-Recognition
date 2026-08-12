"""Tests for the Siamese DVD-cover dataset pipeline."""

import torch

from siamese_dvd.dataset import (
    DVDReferencePairDataset,
    build_baseline_transform,
)


REFERENCE_DIR = "data/reference"


def test_reference_dataset_contains_100_images():
    """The preserved Stanford reference set contains 100 DVD identities."""

    dataset = DVDReferencePairDataset(
        reference_dir=REFERENCE_DIR,
        transform=build_baseline_transform(),
    )

    assert len(dataset.image_paths) == 100
    assert len(dataset) == 800


def test_positive_pair_shape_and_label():
    """Positive samples should produce two valid images with label 0."""

    dataset = DVDReferencePairDataset(
        reference_dir=REFERENCE_DIR,
        transform=build_baseline_transform(),
        pairs_per_epoch=10,
        positive_probability=1.0,
    )

    image1, image2, label = dataset[0]

    assert image1.shape == (1, 100, 100)
    assert image2.shape == (1, 100, 100)
    assert image1.dtype == torch.float32
    assert image2.dtype == torch.float32
    assert label.shape == (1,)
    assert label.item() == 0.0


def test_negative_pair_shape_and_label():
    """Negative samples should produce two valid images with label 1."""

    dataset = DVDReferencePairDataset(
        reference_dir=REFERENCE_DIR,
        transform=build_baseline_transform(),
        pairs_per_epoch=10,
        positive_probability=0.0,
    )

    image1, image2, label = dataset[0]

    assert image1.shape == (1, 100, 100)
    assert image2.shape == (1, 100, 100)
    assert label.shape == (1,)
    assert label.item() == 1.0