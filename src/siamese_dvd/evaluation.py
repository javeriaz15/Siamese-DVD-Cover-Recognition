"""Evaluation dataset utilities for real DVD-cover query images."""

from pathlib import Path

from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DVDEvaluationDataset(Dataset):
    """Dataset of real device-captured DVD-cover query images.

    Each filename corresponds directly to its reference identity.

    Example:
        Canon/017.jpg -> reference identity "017"
    """

    def __init__(
        self,
        dataset_root: str | Path,
        devices: tuple[str, ...] = (
            "Canon",
            "Droid",
            "E63",
            "Palm",
        ),
    ):
        self.dataset_root = Path(dataset_root)
        self.devices = devices

        self.samples: list[tuple[Path, str, str]] = []

        for device in self.devices:
            device_dir = self.dataset_root / device

            if not device_dir.exists():
                raise FileNotFoundError(
                    f"Evaluation directory not found: {device_dir}"
                )

            image_paths = sorted(
                path
                for path in device_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            for path in image_paths:
                reference_id = path.stem

                self.samples.append(
                    (
                        path,
                        reference_id,
                        device,
                    )
                )

        if not self.samples:
            raise ValueError(
                "No evaluation images were found."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Path, str, str]:
        """Return image path, reference identity, and source device."""

        return self.samples[index]