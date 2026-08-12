"""Dataset utilities for Siamese DVD-cover metric learning."""

import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class RandomDiscreteRotation:
    """Rotate an image by one of the angles used in the original project."""

    def __init__(
        self,
        angles: Sequence[int] = (0, 45, 90, 135, 180, 225, 270, 315),
    ):
        self.angles = tuple(angles)

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)

        return TF.rotate(
            image,
            angle=angle,
            expand=True,
            fill=0,
        )


def build_baseline_transform(image_size: int = 100):
    """Build preprocessing compatible with the baseline Siamese network."""

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            RandomDiscreteRotation(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

def build_inference_transform(image_size: int = 100):
    """Build deterministic preprocessing for inference and retrieval."""

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

class DVDReferencePairDataset(Dataset):
    """Generate similar and dissimilar DVD-cover pairs on demand.

    The reference dataset contains one canonical image per DVD identity.

    Positive pair:
        Two independently augmented views of the same DVD cover.

    Negative pair:
        Augmented views of two different DVD covers.

    Label convention:
        0 = similar
        1 = dissimilar
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        reference_dir: str | Path,
        transform=None,
        pairs_per_epoch: int = 800,
        positive_probability: float = 0.5,
    ):
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.pairs_per_epoch = pairs_per_epoch
        self.positive_probability = positive_probability

        self.image_paths = sorted(
            path
            for path in self.reference_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if len(self.image_paths) < 2:
            raise ValueError(
                "Reference directory must contain at least two images."
            )

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del index

        anchor_path = random.choice(self.image_paths)

        is_positive = random.random() < self.positive_probability

        if is_positive:
            pair_path = anchor_path
            label = 0.0
        else:
            pair_path = random.choice(self.image_paths)

            while pair_path == anchor_path:
                pair_path = random.choice(self.image_paths)

            label = 1.0

        image1 = self._load_image(anchor_path)
        image2 = self._load_image(pair_path)

        if self.transform is not None:
            # Apply independently so positive pairs can receive
            # different augmentations.
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image1, image2, label_tensor