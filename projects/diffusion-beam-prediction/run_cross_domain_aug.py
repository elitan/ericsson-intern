#!/usr/bin/env python3
"""Cross-domain experiment with traditional augmentation."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

from src.diffbeam.dataset import DeepSenseDataset
from src.diffbeam.models import create_model
from src.diffbeam.evaluate import evaluate_model


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper that adds augmentation to a dataset."""

    def __init__(self, base_dataset, augment=True):
        self.base = base_dataset
        self.augment = augment

        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.augment:
            img = sample["image"]
            img = self.aug_transform(img)
            sample = dict(sample)
            sample["image"] = img
        return sample


def create_cross_domain_loaders_aug(
    data_dir: str,
    train_scenarios: list[int],
    test_scenarios: list[int],
    batch_size: int = 32,
    num_workers: int = 4,
    use_aug: bool = True,
):
    """Create dataloaders with optional augmentation."""
    train_dataset = DeepSenseDataset(
        data_dir,
        scenarios=train_scenarios,
        split="train",
        train_ratio=1.0,
    )

    if use_aug:
        train_dataset = AugmentedDataset(train_dataset, augment=True)

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
    parser.add_argument("--train-scenarios", type=int, nargs="+", default=[31, 32])
    parser.add_argument("--test-scenarios", type=int, nargs="+", default=[33, 34])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-aug", action="store_true", help="Disable augmentation")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    use_aug = not args.no_aug
    print(f"\n=== Cross-Domain + {'Traditional Aug' if use_aug else 'No Aug'} ===")
    print(f"Train scenarios: {args.train_scenarios} (day)")
    print(f"Test scenarios: {args.test_scenarios} (night)")
    print(f"Augmentation: {use_aug}")

    train_loader, test_loader = create_cross_domain_loaders_aug(
        args.data_dir,
        train_scenarios=args.train_scenarios,
        test_scenarios=args.test_scenarios,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=use_aug,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = create_model("multimodal", num_beams=64, pretrained=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        test_metrics = evaluate_model(model, test_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test (night) Top-1: {test_metrics['top_1_accuracy']:.2%}")
        print(f"Test (night) Top-3: {test_metrics['top_3_accuracy']:.2%}")
        print(f"Test (night) Top-5: {test_metrics['top_5_accuracy']:.2%}")

        if test_metrics["top_1_accuracy"] > best_acc:
            best_acc = test_metrics["top_1_accuracy"]
            suffix = "_aug" if use_aug else "_noaug"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "augmentation": use_aug,
            }, save_dir / f"cross_domain{suffix}.pt")
            print(f"Saved best model (top-1: {best_acc:.2%})")

    print(f"\n=== Final Results ===")
    print(f"Augmentation: {use_aug}")
    print(f"Best Top-1: {best_acc:.2%}")


if __name__ == "__main__":
    main()
