#!/usr/bin/env python3
"""Run full experiment suite for diffusion-augmented beam prediction."""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.diffbeam.dataset import create_dataloaders
from src.diffbeam.models import create_model
from src.diffbeam.diffusion import BeamConditionedDiffusion, DiffusionAugmenter, TraditionalAugmenter
from src.diffbeam.evaluate import evaluate_model, MetricsTracker


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    augmenter=None,
    aug_ratio: float = 0.0,
    traditional_aug: bool = False,
) -> dict:
    """Train model and return results."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    trad_aug = TraditionalAugmenter() if traditional_aug else None

    tracker = MetricsTracker()
    best_acc = 0
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images = batch["image"].to(device)
            gps = batch["gps"].to(device)
            targets = batch["beam_index"].to(device)

            if trad_aug is not None:
                images = trad_aug(images).to(device)

            optimizer.zero_grad()
            logits = model(images, gps)
            loss = criterion(logits, targets)

            if augmenter is not None and aug_ratio > 0:
                num_aug = int(images.size(0) * aug_ratio)
                if num_aug > 0:
                    aug_indices = torch.randint(0, 64, (num_aug,))
                    aug_images, aug_gps = augmenter.augment_batch(aug_indices, gps[:num_aug])
                    aug_images = aug_images.to(device)
                    aug_gps = aug_gps.to(device)
                    aug_targets = aug_indices.to(device)
                    aug_logits = model(aug_images, aug_gps)
                    loss = loss + criterion(aug_logits, aug_targets) * 0.5

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        val_metrics = evaluate_model(model, val_loader, device)
        tracker.update(val_metrics)
        scheduler.step()

        if val_metrics["top_1_accuracy"] > best_acc:
            best_acc = val_metrics["top_1_accuracy"]
            best_metrics = val_metrics.copy()

    return {
        "best_top_1": best_acc,
        "best_top_3": best_metrics.get("top_3_accuracy", 0),
        "best_top_5": best_metrics.get("top_5_accuracy", 0),
        "history": tracker.history,
    }


def run_experiments(args):
    """Run all experiments."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        use_synthetic=args.synthetic,
    )

    results = {}

    print("\n" + "="*60)
    print("Experiment 1: Baseline Models")
    print("="*60)

    for model_type in ["gps_only", "image_only", "multimodal"]:
        print(f"\n--- {model_type} ---")
        model = create_model(model_type, num_beams=64).to(device)
        result = train_model(model, train_loader, val_loader, device, epochs=args.epochs)
        results[f"baseline_{model_type}"] = result
        print(f"Top-1: {result['best_top_1']:.2%}, Top-3: {result['best_top_3']:.2%}")

    print("\n" + "="*60)
    print("Experiment 2: Traditional Augmentation")
    print("="*60)

    model = create_model("multimodal", num_beams=64).to(device)
    result = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, traditional_aug=True
    )
    results["traditional_aug"] = result
    print(f"Top-1: {result['best_top_1']:.2%}, Top-3: {result['best_top_3']:.2%}")

    print("\n" + "="*60)
    print("Experiment 3: Train Diffusion Model")
    print("="*60)

    diffusion = BeamConditionedDiffusion(image_size=64, num_beams=64, device=str(device))

    print("Training diffusion model...")
    resize = torch.nn.functional.interpolate
    optimizer = AdamW(diffusion.model.parameters(), lr=1e-4)

    for epoch in range(args.diffusion_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Diffusion {epoch+1}", leave=False):
            images = batch["image"]
            beam_indices = batch["beam_index"]
            images_resized = resize(images, size=(64, 64), mode="bilinear")
            images_resized = (images_resized - 0.5) / 0.5
            loss = diffusion.train_step(images_resized, beam_indices, optimizer)
            total_loss += loss
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    diffusion.save(save_dir)

    print("\n" + "="*60)
    print("Experiment 4: Diffusion Augmentation Ablation")
    print("="*60)

    augmenter = DiffusionAugmenter(diffusion, target_size=224)

    for aug_ratio in [0.5, 1.0, 2.0, 5.0]:
        print(f"\n--- Aug Ratio: {aug_ratio}x ---")
        model = create_model("multimodal", num_beams=64).to(device)
        result = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, augmenter=augmenter, aug_ratio=aug_ratio
        )
        results[f"diffusion_aug_{aug_ratio}x"] = result
        print(f"Top-1: {result['best_top_1']:.2%}, Top-3: {result['best_top_3']:.2%}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Experiment':<30} {'Top-1':>10} {'Top-3':>10}")
    print("-"*50)
    for name, res in results.items():
        print(f"{name:<30} {res['best_top_1']:>10.2%} {res['best_top_3']:>10.2%}")

    results_file = save_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "history"}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--diffusion-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    run_experiments(args)


if __name__ == "__main__":
    main()
