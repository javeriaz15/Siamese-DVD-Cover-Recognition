"""Run DVD-cover retrieval against the reference gallery."""

import argparse
from pathlib import Path

import torch

from siamese_dvd import (
    BaselineSiameseNetwork,
    build_inference_transform,
    build_reference_gallery,
    encode_image,
    search_gallery,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Identify a DVD cover by retrieving the closest "
            "matches from the reference gallery."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the query DVD-cover image.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained baseline model weights.",
    )

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/reference"),
        help="Directory containing reference DVD covers.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest matches to return.",
    )

    return parser.parse_args()


def get_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(
    weights_path: Path,
    device: torch.device,
) -> BaselineSiameseNetwork:
    """Load the trained baseline Siamese model."""

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    model = BaselineSiameseNetwork().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(
            f"Query image not found: {args.image}"
        )

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    device = get_device()

    print(f"Device: {device}")
    print(f"Query image: {args.image}")

    model = load_model(
        weights_path=args.weights,
        device=device,
    )

    transform = build_inference_transform()

    print("Building reference gallery...")

    gallery_embeddings, gallery_paths = (
        build_reference_gallery(
            model=model,
            reference_dir=args.reference_dir,
            transform=transform,
            device=device,
        )
    )

    query_embedding = encode_image(
        model=model,
        image_path=args.image,
        transform=transform,
        device=device,
    )

    results = search_gallery(
        query_embedding=query_embedding,
        gallery_embeddings=gallery_embeddings,
        gallery_paths=gallery_paths,
        top_k=args.top_k,
    )

    print("\nTop matches:")

    for result in results:
        print(
            f"{result['rank']:>2}. "
            f"DVD {result['reference_id']} "
            f"| distance={result['distance']:.6f}"
        )


if __name__ == "__main__":
    main()