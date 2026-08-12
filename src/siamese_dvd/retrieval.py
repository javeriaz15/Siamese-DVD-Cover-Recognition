"""Embedding-based retrieval utilities for DVD-cover recognition."""

from pathlib import Path

import torch
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image as RGB."""

    with Image.open(image_path) as image:
        return image.convert("RGB")


@torch.no_grad()
def encode_image(
    model,
    image_path: str | Path,
    transform,
    device: torch.device,
) -> torch.Tensor:
    """Encode one image into a Siamese embedding."""

    image = load_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    embedding = model.encode(tensor)

    return embedding


@torch.no_grad()
def build_reference_gallery(
    model,
    reference_dir: str | Path,
    transform,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[torch.Tensor, list[Path]]:
    """Encode all reference DVD covers into a searchable gallery."""

    reference_dir = Path(reference_dir)

    image_paths = sorted(
        path
        for path in reference_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in {reference_dir}"
        )

    model.eval()

    gallery_embeddings = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[
            start : start + batch_size
        ]

        batch_tensors = [
            transform(load_image(path))
            for path in batch_paths
        ]

        batch = torch.stack(
            batch_tensors
        ).to(device)

        embeddings = model.encode(batch)

        gallery_embeddings.append(
            embeddings.cpu()
        )

    gallery = torch.cat(
        gallery_embeddings,
        dim=0,
    )

    return gallery, image_paths


def search_gallery(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_paths: list[Path],
    top_k: int = 5,
) -> list[dict]:
    """Return the nearest reference images by Euclidean distance."""

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)

    query_embedding = query_embedding.cpu()

    distances = torch.cdist(
        query_embedding,
        gallery_embeddings,
        p=2,
    ).squeeze(0)

    top_k = min(
        top_k,
        len(gallery_paths),
    )

    values, indices = torch.topk(
        distances,
        k=top_k,
        largest=False,
    )

    results = []

    for rank, (distance, index) in enumerate(
        zip(values, indices),
        start=1,
    ):
        path = gallery_paths[index.item()]

        results.append(
            {
                "rank": rank,
                "reference_id": path.stem,
                "path": str(path),
                "distance": distance.item(),
            }
        )

    return results