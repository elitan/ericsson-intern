#!/usr/bin/env python3
"""Cross-domain with offline diffusion augmentation.

Strategy:
1. Train diffusion model on day images (once)
2. Pre-generate synthetic night-style images (offline)
3. Train beam predictor on real + synthetic data (fast)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np

from src.diffbeam.dataset import DeepSenseDataset
from src.diffbeam.models import create_model
from src.diffbeam.diffusion import BeamConditionedDiffusion
from src.diffbeam.evaluate import evaluate_model


class SyntheticAugDataset(Dataset):
    """Dataset of pre-generated synthetic images."""

    def __init__(self, images_path: Path):
        data = torch.load(images_path)
        self.images = data["images"]
        self.gps = data["gps"]
        self.beam_indices = data["beam_indices"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "gps": self.gps[idx],
            "beam_index": self.beam_indices[idx],
            "scenario": -1,
        }


def generate_synthetic_dataset(
    diffusion: BeamConditionedDiffusion,
    train_loader,
    num_samples: int,
    save_path: Path,
    num_inference_steps: int = 20,
):
    """Pre-generate synthetic images for augmentation."""
    print(f"\n=== Generating {num_samples} Synthetic Images ===")

    all_images = []
    all_gps = []
    all_beams = []

    generated = 0
    pbar = tqdm(total=num_samples, desc="Generating")

    while generated < num_samples:
        for batch in train_loader:
            if generated >= num_samples:
                break

            beam_indices = batch["beam_index"]
            gps = batch["gps"]

            batch_size = min(len(beam_indices), num_samples - generated)
            beam_indices = beam_indices[:batch_size]
            gps = gps[:batch_size]

            with torch.no_grad():
                images = diffusion.generate(beam_indices, num_inference_steps=num_inference_steps)

            from torchvision import transforms
            resize = transforms.Resize((224, 224))
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            images = resize(images)
            images = normalize(images)

            gps_noise = torch.randn_like(gps) * 0.1
            gps_perturbed = gps + gps_noise

            all_images.append(images.cpu())
            all_gps.append(gps_perturbed)
            all_beams.append(beam_indices)

            generated += batch_size
            pbar.update(batch_size)

    pbar.close()

    images_tensor = torch.cat(all_images, dim=0)[:num_samples]
    gps_tensor = torch.cat(all_gps, dim=0)[:num_samples]
    beams_tensor = torch.cat(all_beams, dim=0)[:num_samples]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "images": images_tensor,
        "gps": gps_tensor,
        "beam_indices": beams_tensor,
    }, save_path)

    print(f"Saved {len(images_tensor)} synthetic images to {save_path}")
    return save_path


def train_diffusion(
    diffusion: BeamConditionedDiffusion,
    dataloader,
    epochs: int = 30,
    lr: float = 1e-4,
    save_dir: Path = None,
):
    print("\n=== Training Diffusion Model ===")

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

            images_resized = resize(images, size=(target_size, target_size), mode="bilinear")
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

    return diffusion


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
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader.dataset)


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
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--inference-steps", type=int, default=20)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"\n=== Cross-Domain + Offline Diffusion Augmentation ===")
    print(f"Train scenarios: {args.train_scenarios} (day)")
    print(f"Test scenarios: {args.test_scenarios} (night)")
    print(f"Aug ratio: {args.aug_ratio}")

    train_dataset = DeepSenseDataset(
        args.data_dir,
        scenarios=args.train_scenarios,
        split="train",
        train_ratio=1.0,
    )

    test_dataset = DeepSenseDataset(
        args.data_dir,
        scenarios=args.test_scenarios,
        split="train",
        train_ratio=1.0,
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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    save_dir = Path(args.save_dir) / "cross_domain_diffusion_offline"
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

    synthetic_path = save_dir / "synthetic_data.pt"
    num_synthetic = int(len(train_dataset) * args.aug_ratio)

    if args.skip_generation and synthetic_path.exists():
        print(f"Loading pre-generated synthetic data from {synthetic_path}")
    else:
        generate_synthetic_dataset(
            diffusion, train_loader,
            num_samples=num_synthetic,
            save_path=synthetic_path,
            num_inference_steps=args.inference_steps,
        )

    synthetic_dataset = SyntheticAugDataset(synthetic_path)
    combined_dataset = ConcatDataset([train_dataset, synthetic_dataset])

    combined_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(f"\n=== Training Beam Predictor ===")
    print(f"Real samples: {len(train_dataset)}")
    print(f"Synthetic samples: {len(synthetic_dataset)}")
    print(f"Total: {len(combined_dataset)}")

    model = create_model("multimodal", num_beams=64, pretrained=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, combined_loader, optimizer, criterion, device)

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
    print(f"Method: Offline Diffusion Augmentation (ratio={args.aug_ratio})")
    print(f"Best Top-1: {best_acc:.2%}")


if __name__ == "__main__":
    main()
