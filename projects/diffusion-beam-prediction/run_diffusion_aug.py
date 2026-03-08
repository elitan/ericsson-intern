#!/usr/bin/env python3
"""Train diffusion model and beam predictor with augmentation."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.diffbeam.dataset import create_dataloaders, SyntheticDeepSenseDataset
from src.diffbeam.models import create_model
from src.diffbeam.diffusion import BeamConditionedDiffusion, DiffusionAugmenter
from src.diffbeam.evaluate import evaluate_model, MetricsTracker


def train_diffusion(
    diffusion: BeamConditionedDiffusion,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
) -> BeamConditionedDiffusion:
    """Train diffusion model on dataset images."""
    print("\n=== Training Diffusion Model ===")

    resize = torch.nn.functional.interpolate
    target_size = diffusion.image_size

    optimizer = AdamW(diffusion.model.parameters(), lr=lr)

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
            pbar.set_postfix({"loss": loss})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

    return diffusion


def train_with_augmentation(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    augmenter: DiffusionAugmenter,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    aug_ratio: float = 1.0,
) -> tuple[nn.Module, MetricsTracker]:
    """Train beam predictor with diffusion augmentation."""
    print(f"\n=== Training Beam Predictor (aug_ratio={aug_ratio}) ===")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    tracker = MetricsTracker()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            gps = batch["gps"].to(device)
            targets = batch["beam_index"].to(device)

            optimizer.zero_grad()
            logits = model(images, gps)
            loss = criterion(logits, targets)

            if aug_ratio > 0 and augmenter is not None:
                num_aug = int(images.size(0) * aug_ratio)
                if num_aug > 0:
                    aug_indices = torch.randint(0, 64, (num_aug,))
                    aug_images, aug_gps = augmenter.augment_batch(
                        aug_indices, gps[:num_aug]
                    )
                    aug_images = aug_images.to(device)
                    aug_gps = aug_gps.to(device)
                    aug_targets = aug_indices.to(device)

                    aug_logits = model(aug_images, aug_gps)
                    aug_loss = criterion(aug_logits, aug_targets)
                    loss = loss + aug_loss * 0.5

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = total_loss / num_samples
        tracker.update({"loss": train_loss}, prefix="train")

        val_metrics = evaluate_model(model, val_loader, device)
        tracker.update(val_metrics)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | "
              f"Val Top-1: {val_metrics['top_1_accuracy']:.2%} | "
              f"Top-3: {val_metrics['top_3_accuracy']:.2%}")

        if val_metrics["top_1_accuracy"] > best_acc:
            best_acc = val_metrics["top_1_accuracy"]

    return model, tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--diffusion-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aug-ratio", type=float, default=1.0,
                        help="Ratio of augmented samples per real sample")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--skip-diffusion-train", action="store_true",
                        help="Load pretrained diffusion model")
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
        num_workers=4,
        use_synthetic=args.synthetic,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    diffusion = BeamConditionedDiffusion(
        image_size=64,
        num_beams=64,
        device=str(device),
    )

    if args.skip_diffusion_train and (save_dir / "diffusion_model.pt").exists():
        print("Loading pretrained diffusion model...")
        diffusion.load(save_dir)
    else:
        diffusion = train_diffusion(
            diffusion, train_loader,
            epochs=args.diffusion_epochs,
            lr=1e-4,
        )
        diffusion.save(save_dir)

    augmenter = DiffusionAugmenter(diffusion, target_size=224)

    model = create_model("multimodal", num_beams=64, pretrained=True)
    model = model.to(device)

    model, tracker = train_with_augmentation(
        model, train_loader, val_loader,
        augmenter=augmenter,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        aug_ratio=args.aug_ratio,
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "aug_ratio": args.aug_ratio,
        "history": tracker.history,
    }, save_dir / "augmented_model.pt")

    best_epoch, best_acc = tracker.get_best("top_1_accuracy")
    print(f"\nBest Top-1 Accuracy: {best_acc:.2%} (epoch {best_epoch + 1})")


if __name__ == "__main__":
    main()
