"""Train the baseline Siamese network for DVD-cover metric learning."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from siamese_dvd import (
    BaselineSiameseNetwork,
    ContrastiveLoss,
    DVDReferencePairDataset,
    build_baseline_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the baseline Siamese DVD-cover recognition model."
    )

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/reference"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--pairs-per-epoch",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("artifacts/weights/baseline_weights.pt"),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA when available, otherwise use CPU."""

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def main():
    args = parse_args()

    set_seed(args.seed)

    device = get_device()

    print(f"Device: {device}")
    print(f"Reference directory: {args.reference_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Pairs per epoch: {args.pairs_per_epoch}")

    dataset = DVDReferencePairDataset(
        reference_dir=args.reference_dir,
        transform=build_baseline_transform(),
        pairs_per_epoch=args.pairs_per_epoch,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = BaselineSiameseNetwork().to(device)

    criterion = ContrastiveLoss(
        margin=args.margin
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        samples_seen = 0

        for image1, image2, labels in dataloader:
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            embedding1, embedding2 = model(
                image1,
                image2,
            )

            loss = criterion(
                embedding1,
                embedding2,
                labels,
            )

            loss.backward()
            optimizer.step()

            batch_size = image1.size(0)

            running_loss += loss.item() * batch_size
            samples_seen += batch_size

        average_loss = running_loss / samples_seen

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| loss={average_loss:.6f}"
        )

    args.weights_path.parent.mkdir(
        parents=True,
        exist_ok=True,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "pairs_per_epoch": args.pairs_per_epoch,
            "margin": args.margin,
            "seed": args.seed,
        },
        args.weights_path,
    )

    print(f"Model weights saved to: {args.weights_path}")


if __name__ == "__main__":
    main()