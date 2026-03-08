#!/usr/bin/env python3
"""GPS-only baseline: Does vision even help cross-domain?

Compares:
1. GPS-only same-domain (train/test all scenarios)
2. GPS-only cross-domain (train day, test night)
3. Multimodal cross-domain (from run_cross_domain.py)

If GPS-only cross-domain ≈ multimodal cross-domain, vision isn't helping.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.diffbeam.dataset import DeepSenseDataset
from src.diffbeam.models import GPSOnlyPredictor


def create_dataloaders(
    data_dir: str,
    train_scenarios: list[int],
    test_scenarios: list[int],
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Create train/test loaders."""
    train_dataset = DeepSenseDataset(
        data_dir,
        scenarios=train_scenarios,
        split="train",
        train_ratio=1.0,
    )
    test_dataset = DeepSenseDataset(
        data_dir,
        scenarios=test_scenarios,
        split="train",
        train_ratio=1.0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        gps = batch["gps"].to(device)
        targets = batch["beam_index"].to(device)

        optimizer.zero_grad()
        logits = model(image=None, gps=gps)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * gps.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            gps = batch["gps"].to(device)
            targets = batch["beam_index"].to(device)

            logits = model(image=None, gps=gps)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    results = {}
    for k in [1, 3, 5, 10]:
        _, top_k = torch.topk(all_logits, k, dim=1)
        correct = (top_k == all_targets.unsqueeze(1)).any(dim=1)
        results[f"top_{k}"] = correct.float().mean().item()

    return results


def run_experiment(name, data_dir, train_scenarios, test_scenarios, epochs, batch_size, lr, device):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Train: {train_scenarios}, Test: {test_scenarios}")
    print(f"{'='*60}")

    train_loader, test_loader = create_dataloaders(
        data_dir, train_scenarios, test_scenarios, batch_size
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = GPSOnlyPredictor(num_beams=64).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_results = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_results = evaluate(model, test_loader, device)
        scheduler.step()

        if test_results["top_1"] > best_acc:
            best_acc = test_results["top_1"]
            best_results = test_results.copy()

        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f} | "
              f"Top-1={test_results['top_1']:.2%} | "
              f"Top-3={test_results['top_3']:.2%} | "
              f"Top-5={test_results['top_5']:.2%}")

    print(f"\nBest {name}:")
    for k, v in best_results.items():
        print(f"  {k}: {v:.2%}")

    return best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    results = {}

    results["same_domain"] = run_experiment(
        name="GPS-only Same-Domain",
        data_dir=args.data_dir,
        train_scenarios=[31, 32, 33, 34],
        test_scenarios=[31, 32, 33, 34],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    results["cross_domain"] = run_experiment(
        name="GPS-only Cross-Domain (Day→Night)",
        data_dir=args.data_dir,
        train_scenarios=[31, 32],
        test_scenarios=[33, 34],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    print("\n" + "="*60)
    print("SUMMARY: GPS-Only Baseline")
    print("="*60)
    print(f"{'Setup':<30} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10}")
    print("-"*60)
    print(f"{'Same-domain (all)':<30} {results['same_domain']['top_1']:>10.2%} "
          f"{results['same_domain']['top_3']:>10.2%} {results['same_domain']['top_5']:>10.2%}")
    print(f"{'Cross-domain (day→night)':<30} {results['cross_domain']['top_1']:>10.2%} "
          f"{results['cross_domain']['top_3']:>10.2%} {results['cross_domain']['top_5']:>10.2%}")
    print("-"*60)
    print(f"\nFor reference (from README):")
    print(f"  Multimodal same-domain:  45.47% top-1")
    print(f"  Multimodal cross-domain:  4.67% top-1")
    print(f"\nConclusion:")
    if results['cross_domain']['top_1'] > 0.10:
        print(f"  GPS alone achieves {results['cross_domain']['top_1']:.1%} cross-domain.")
        print(f"  Vision likely adds only {4.67 - results['cross_domain']['top_1']*100:.1f}pp (or hurts)")
    else:
        print(f"  GPS alone also fails cross-domain ({results['cross_domain']['top_1']:.1%})")
        print(f"  Both modalities need domain adaptation")


if __name__ == "__main__":
    main()
