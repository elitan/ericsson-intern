#!/usr/bin/env python3
"""Oracle baseline: Train on night, test on night.

Shows the ceiling for night domain performance.
Compare to cross-domain 4.67% to see the gap we're trying to close.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.diffbeam.dataset import DeepSenseDataset
from src.diffbeam.models import create_model
from src.diffbeam.evaluate import evaluate_model


def train_epoch(model, dataloader, optimizer, criterion, device):
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("\n=== Oracle Experiment: Train & Test on Night (33+34) ===")

    train_dataset = DeepSenseDataset(
        args.data_dir,
        scenarios=[33, 34],
        split="train",
        train_ratio=args.train_ratio,
    )
    test_dataset = DeepSenseDataset(
        args.data_dir,
        scenarios=[33, 34],
        split="val",
        train_ratio=args.train_ratio,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train samples (night): {len(train_dataset)}")
    print(f"Test samples (night): {len(test_dataset)}")

    model = create_model("multimodal", num_beams=64, pretrained=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_metrics = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate_model(model, test_loader, device)
        scheduler.step()

        if test_metrics["top_1_accuracy"] > best_acc:
            best_acc = test_metrics["top_1_accuracy"]
            best_metrics = test_metrics.copy()

        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f} | "
              f"Top-1={test_metrics['top_1_accuracy']:.2%} | "
              f"Top-3={test_metrics['top_3_accuracy']:.2%} | "
              f"Top-5={test_metrics['top_5_accuracy']:.2%}")

    print("\n" + "="*60)
    print("ORACLE RESULTS (Night-only training)")
    print("="*60)
    print(f"Best Top-1: {best_metrics['top_1_accuracy']:.2%}")
    print(f"Best Top-3: {best_metrics['top_3_accuracy']:.2%}")
    print(f"Best Top-5: {best_metrics['top_5_accuracy']:.2%}")
    print("-"*60)
    print("For reference:")
    print(f"  Same-domain (all scenarios):    45.47% top-1")
    print(f"  Cross-domain (day→night):        4.67% top-1")
    print(f"  Cross-domain + diffusion:        6.26% top-1")
    print("-"*60)
    gap_to_close = best_metrics['top_1_accuracy'] - 0.0467
    gap_closed_diff = 0.0626 - 0.0467
    print(f"\nDomain gap to close: {gap_to_close:.1%}")
    print(f"Gap closed by diffusion: {gap_closed_diff:.1%} ({gap_closed_diff/gap_to_close*100:.1f}% of gap)")


if __name__ == "__main__":
    main()
