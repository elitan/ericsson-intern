#!/usr/bin/env python3
"""Cross-domain experiment with diffusion augmentation.

Train diffusion on DAY (31+32), use it to augment training, test on NIGHT (33+34).
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
from src.diffbeam.diffusion import BeamConditionedDiffusion, DiffusionAugmenter
from src.diffbeam.evaluate import evaluate_model


def create_cross_domain_loaders(
    data_dir: str,
    train_scenarios: list[int],
    test_scenarios: list[int],
    batch_size: int = 32,
    num_workers: int = 4,
):
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


def train_diffusion(
    diffusion: BeamConditionedDiffusion,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 30,
    lr: float = 1e-4,
    save_dir: Path = None,
) -> BeamConditionedDiffusion:
    print("\n=== Training Diffusion Model on Day Images ===")

    resize = torch.nn.functional.interpolate
    target_size = diffusion.image_size

    optimizer = AdamW(diffusion.model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["image"]
            beam_indices = batch["beam_index"]

            images_resized = resize(
                images, size=(target_size, target_size), mode="bilinear"
            )
            images_resized = (images_resized - 0.5) / 0.5

            loss = diffusion.train_step(images_resized, beam_indices, optimizer)
            total_loss += loss
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

        if avg_loss < best_loss and save_dir:
            best_loss = avg_loss
            diffusion.save(save_dir)
            print(f"  Saved best diffusion model (loss: {best_loss:.4f})")

    return diffusion


def train_epoch_with_aug(
    model, dataloader, optimizer, criterion, device,
    augmenter=None, aug_ratio=1.0
):
    model.train()
    total_loss = 0
    num_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_index"].to(device)

        optimizer.zero_grad()
        logits = model(images, gps)
        loss = criterion(logits, targets)

        if augmenter is not None and aug_ratio > 0:
            num_aug = int(images.size(0) * aug_ratio)
            if num_aug > 0:
                aug_images, aug_gps = augmenter.augment_batch(
                    targets[:num_aug], gps[:num_aug]
                )
                aug_images = aug_images.to(device)
                aug_gps = aug_gps.to(device)

                aug_logits = model(aug_images, aug_gps)
                aug_loss = criterion(aug_logits, targets[:num_aug])
                loss = loss + aug_loss * 0.5

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--train-scenarios", type=int, nargs="+", default=[31, 32])
    parser.add_argument("--test-scenarios", type=int, nargs="+", default=[33, 34])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--diffusion-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aug-ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--skip-diffusion", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"\n=== Cross-Domain + Diffusion Augmentation ===")
    print(f"Train scenarios: {args.train_scenarios} (day)")
    print(f"Test scenarios: {args.test_scenarios} (night)")
    print(f"Aug ratio: {args.aug_ratio}")

    train_loader, test_loader = create_cross_domain_loaders(
        args.data_dir,
        train_scenarios=args.train_scenarios,
        test_scenarios=args.test_scenarios,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    save_dir = Path(args.save_dir) / "cross_domain_diffusion"
    save_dir.mkdir(parents=True, exist_ok=True)

    diffusion = BeamConditionedDiffusion(
        image_size=64,
        num_beams=64,
        device=str(device),
    )

    diffusion_path = save_dir / "diffusion_model.pt"
    if args.skip_diffusion and diffusion_path.exists():
        print("Loading pretrained diffusion model...")
        diffusion.load(save_dir)
    else:
        diffusion = train_diffusion(
            diffusion, train_loader,
            epochs=args.diffusion_epochs,
            lr=1e-4,
            save_dir=save_dir,
        )

    augmenter = DiffusionAugmenter(diffusion, target_size=224)

    print("\n=== Training Beam Predictor with Diffusion Aug ===")
    model = create_model("multimodal", num_beams=64, pretrained=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch_with_aug(
            model, train_loader, optimizer, criterion, device,
            augmenter=augmenter, aug_ratio=args.aug_ratio
        )

        test_metrics = evaluate_model(model, test_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test (night) Top-1: {test_metrics['top_1_accuracy']:.2%}")
        print(f"Test (night) Top-3: {test_metrics['top_3_accuracy']:.2%}")
        print(f"Test (night) Top-5: {test_metrics['top_5_accuracy']:.2%}")

        if test_metrics["top_1_accuracy"] > best_acc:
            best_acc = test_metrics["top_1_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "aug_ratio": args.aug_ratio,
            }, save_dir / "beam_predictor.pt")
            print(f"Saved best model (top-1: {best_acc:.2%})")

    print(f"\n=== Final Results ===")
    print(f"Method: Diffusion Augmentation (ratio={args.aug_ratio})")
    print(f"Best Top-1: {best_acc:.2%}")


if __name__ == "__main__":
    main()
