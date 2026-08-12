"""Tests for the real DVD-cover evaluation dataset."""

from pathlib import Path

import pytest

from siamese_dvd.evaluation import DVDEvaluationDataset


DATASET_ROOT = Path("data/raw/dvd_covers")

pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.exists(),
    reason="Stanford DVD-cover evaluation dataset is not downloaded.",
)


def test_evaluation_dataset_contains_400_queries():
    dataset = DVDEvaluationDataset(
        dataset_root=DATASET_ROOT,
    )

    assert len(dataset) == 400


def test_each_device_contains_100_queries():
    dataset = DVDEvaluationDataset(
        dataset_root=DATASET_ROOT,
    )

    counts = {}

    for _, _, device in dataset:
        counts[device] = counts.get(device, 0) + 1

    assert counts == {
        "Canon": 100,
        "Droid": 100,
        "E63": 100,
        "Palm": 100,
    }


def test_filename_maps_to_reference_identity():
    dataset = DVDEvaluationDataset(
        dataset_root=DATASET_ROOT,
        devices=("Canon",),
    )

    path, reference_id, device = dataset[0]

    assert path.stem == reference_id
    assert device == "Canon"