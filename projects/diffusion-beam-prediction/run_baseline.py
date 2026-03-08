#!/usr/bin/env python3
"""Train baseline beam predictor on DeepSense 6G data."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.diffbeam.dataset import create_dataloaders
from src.diffbeam.models import create_model
from src.diffbeam.evaluate import evaluate_model, MetricsTracker


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_index"].to(device)

        optimizer.zero_grad()
        logits = model(images, gps)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-type", type=str, default="multimodal",
                        choices=["multimodal", "image_only", "gps_only"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_synthetic=args.synthetic,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(args.model_type, num_beams=64, pretrained=True)
    model = model.to(device)
    print(f"Model: {args.model_type}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    tracker = MetricsTracker()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        tracker.update({"loss": train_loss}, prefix="train")

        val_metrics = evaluate_model(model, val_loader, device)
        tracker.update(val_metrics)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Top-1: {val_metrics['top_1_accuracy']:.2%}")
        print(f"Top-3: {val_metrics['top_3_accuracy']:.2%}")
        print(f"Top-5: {val_metrics['top_5_accuracy']:.2%}")

        if val_metrics["top_1_accuracy"] > best_acc:
            best_acc = val_metrics["top_1_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "model_type": args.model_type,
            }, save_dir / "best_model.pt")
            print(f"Saved best model (top-1: {best_acc:.2%})")

    print(f"\nTraining complete. Best top-1 accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
