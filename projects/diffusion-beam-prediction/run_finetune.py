#!/usr/bin/env python3
"""Few-shot fine-tuning: Pretrain on day, fine-tune on N night samples.

Tests data efficiency of transfer learning.
Sweep N = 100, 500, 1000, 2000.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from tqdm import tqdm

from src.diffbeam.dataset import DeepSenseDataset
from src.diffbeam.models import create_model
from src.diffbeam.evaluate import evaluate_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
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


def pretrain_on_day(data_dir, epochs, batch_size, lr, device):
    """Pretrain model on day scenarios (31+32)."""
    print("\n=== Pretraining on Day (31+32) ===")

    train_dataset = DeepSenseDataset(
        data_dir,
        scenarios=[31, 32],
        split="train",
        train_ratio=0.9,
    )
    val_dataset = DeepSenseDataset(
        data_dir,
        scenarios=[31, 32],
        split="val",
        train_ratio=0.9,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Pretrain samples: {len(train_dataset)}")

    model = create_model("multimodal", num_beams=64, pretrained=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        if val_metrics["top_1_accuracy"] > best_acc:
            best_acc = val_metrics["top_1_accuracy"]

        print(f"Pretrain Epoch {epoch+1:2d}: Loss={train_loss:.4f} | "
              f"Val Top-1={val_metrics['top_1_accuracy']:.2%}")

    print(f"Pretrain complete. Best day val: {best_acc:.2%}")
    return model


def finetune_on_night(model, data_dir, n_samples, ft_epochs, batch_size, lr, device, seed=42):
    """Fine-tune pretrained model on N night samples."""
    full_night = DeepSenseDataset(
        data_dir,
        scenarios=[33, 34],
        split="train",
        train_ratio=1.0,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    if n_samples < len(full_night):
        indices = rng.choice(len(full_night), size=n_samples, replace=False)
        train_subset = Subset(full_night, indices)
    else:
        train_subset = full_night
        n_samples = len(full_night)

    test_indices = np.array([i for i in range(len(full_night)) if i not in indices]) if n_samples < len(full_night) else []
    if len(test_indices) > 0:
        test_subset = Subset(full_night, test_indices[:2000])
    else:
        test_subset = DeepSenseDataset(
            data_dir, scenarios=[33, 34], split="val", train_ratio=0.8, seed=seed+1
        )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=min(batch_size, n_samples),
        shuffle=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    ft_model = create_model("multimodal", num_beams=64, pretrained=True)
    ft_model.load_state_dict(model.state_dict())
    ft_model = ft_model.to(device)

    optimizer = AdamW(ft_model.parameters(), lr=lr * 0.1, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_metrics = None

    for epoch in range(ft_epochs):
        train_loss = train_epoch(ft_model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate_model(ft_model, test_loader, device)

        if test_metrics["top_1_accuracy"] > best_acc:
            best_acc = test_metrics["top_1_accuracy"]
            best_metrics = test_metrics.copy()

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-samples", type=int, nargs="+", default=[100, 500, 1000, 2000])
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    pretrained_model = pretrain_on_day(
        args.data_dir,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    night_test = DeepSenseDataset(
        args.data_dir, scenarios=[33, 34], split="train", train_ratio=1.0
    )
    test_loader = torch.utils.data.DataLoader(night_test, batch_size=args.batch_size, num_workers=4)
    zero_shot = evaluate_model(pretrained_model, test_loader, device)
    print(f"\nZero-shot (pretrained, no fine-tune): {zero_shot['top_1_accuracy']:.2%}")

    results = {"zero_shot": zero_shot}

    print("\n=== Few-Shot Fine-Tuning Sweep ===")
    for n in args.n_samples:
        print(f"\nFine-tuning on {n} night samples...")
        metrics = finetune_on_night(
            pretrained_model,
            args.data_dir,
            n_samples=n,
            ft_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        results[f"n_{n}"] = metrics
        print(f"  N={n}: Top-1={metrics['top_1_accuracy']:.2%} | "
              f"Top-3={metrics['top_3_accuracy']:.2%} | "
              f"Top-5={metrics['top_5_accuracy']:.2%}")

    print("\n" + "="*60)
    print("FEW-SHOT FINE-TUNING RESULTS")
    print("="*60)
    print(f"{'N Samples':<15} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10}")
    print("-"*60)
    print(f"{'0 (zero-shot)':<15} {zero_shot['top_1_accuracy']:>10.2%} "
          f"{zero_shot['top_3_accuracy']:>10.2%} {zero_shot['top_5_accuracy']:>10.2%}")
    for n in args.n_samples:
        m = results[f"n_{n}"]
        print(f"{n:<15} {m['top_1_accuracy']:>10.2%} {m['top_3_accuracy']:>10.2%} {m['top_5_accuracy']:>10.2%}")
    print("-"*60)
    print("\nFor reference:")
    print(f"  Cross-domain (no ft):     4.67% top-1")
    print(f"  Cross-domain + diffusion: 6.26% top-1")
    print(f"  Oracle (night-only):      ~40-45% top-1 (TBD)")


if __name__ == "__main__":
    main()
