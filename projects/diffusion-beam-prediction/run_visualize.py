#!/usr/bin/env python3
"""Visualize day vs night domain gap.

Creates:
1. Image grid comparing day (31,32) vs night (33,34) samples
2. t-SNE of ResNet features colored by domain
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.diffbeam.dataset import DeepSenseDataset
from src.diffbeam.models import ImageEncoder


def sample_images(dataset, n_samples=50, seed=42):
    """Sample n random images from dataset."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    samples = []
    for idx in indices:
        sample = dataset[idx]
        img_path = dataset.samples[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        samples.append({
            "image": img,
            "scenario": sample["scenario"],
            "beam": sample["beam_index"].item(),
        })
    return samples


def create_image_grid(day_samples, night_samples, output_path, n_cols=10, n_rows=5):
    """Create side-by-side grid of day vs night images."""
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 2, n_rows * 4 + 1))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j

            if idx < len(day_samples):
                axes[i, j].imshow(day_samples[idx]["image"])
                axes[i, j].set_title(f"B{day_samples[idx]['beam']}", fontsize=8)
            axes[i, j].axis("off")

            if idx < len(night_samples):
                axes[n_rows + i, j].imshow(night_samples[idx]["image"])
                axes[n_rows + i, j].set_title(f"B{night_samples[idx]['beam']}", fontsize=8)
            axes[n_rows + i, j].axis("off")

    fig.text(0.02, 0.75, "Day (31+32)", fontsize=14, rotation=90, va="center")
    fig.text(0.02, 0.25, "Night (33+34)", fontsize=14, rotation=90, va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved image grid to {output_path}")


def extract_features(dataset, encoder, device, n_samples=500, seed=42):
    """Extract ResNet features for t-SNE."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    features = []
    scenarios = []
    beams = []

    encoder.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting features"):
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            feat = encoder(img).cpu().numpy().flatten()
            features.append(feat)
            scenarios.append(sample["scenario"])
            beams.append(sample["beam_index"].item())

    return np.array(features), np.array(scenarios), np.array(beams)


def create_tsne_plot(day_features, day_scenarios, night_features, night_scenarios, output_path):
    """Create t-SNE visualization colored by domain."""
    all_features = np.vstack([day_features, night_features])

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings = tsne.fit_transform(all_features)

    n_day = len(day_features)
    day_emb = embeddings[:n_day]
    night_emb = embeddings[n_day:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(day_emb[:, 0], day_emb[:, 1], c="orange", alpha=0.6, s=20, label="Day (31+32)")
    ax.scatter(night_emb[:, 0], night_emb[:, 1], c="blue", alpha=0.6, s=20, label="Night (33+34)")
    ax.set_title("t-SNE by Domain")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    ax = axes[1]
    scenarios = np.concatenate([day_scenarios, night_scenarios])
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=scenarios, cmap="tab10", alpha=0.6, s=20)
    ax.set_title("t-SNE by Scenario")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Scenario")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")


def create_tsne_by_beam(features, beams, output_path, title=""):
    """Create t-SNE colored by beam index."""
    print("Computing t-SNE by beam...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=beams, cmap="hsv", alpha=0.6, s=20)
    ax.set_title(f"t-SNE by Beam Index {title}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Beam Index")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved beam t-SNE to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/visualize")
    parser.add_argument("--n-grid", type=int, default=50, help="Samples per domain for grid")
    parser.add_argument("--n-tsne", type=int, default=500, help="Samples per domain for t-SNE")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("\n=== Loading Datasets ===")
    day_dataset = DeepSenseDataset(
        args.data_dir,
        scenarios=[31, 32],
        split="train",
        train_ratio=1.0,
    )
    night_dataset = DeepSenseDataset(
        args.data_dir,
        scenarios=[33, 34],
        split="train",
        train_ratio=1.0,
    )
    print(f"Day samples: {len(day_dataset)}")
    print(f"Night samples: {len(night_dataset)}")

    print("\n=== Creating Image Grid ===")
    day_samples = sample_images(day_dataset, n_samples=args.n_grid)
    night_samples = sample_images(night_dataset, n_samples=args.n_grid)
    create_image_grid(day_samples, night_samples, output_dir / "day_vs_night_grid.png")

    print("\n=== Extracting Features for t-SNE ===")
    encoder = ImageEncoder(output_dim=512, pretrained=True).to(device)

    day_features, day_scenarios, day_beams = extract_features(
        day_dataset, encoder, device, n_samples=args.n_tsne
    )
    night_features, night_scenarios, night_beams = extract_features(
        night_dataset, encoder, device, n_samples=args.n_tsne
    )

    print("\n=== Creating t-SNE Plots ===")
    create_tsne_plot(
        day_features, day_scenarios,
        night_features, night_scenarios,
        output_dir / "tsne_domain.png"
    )

    create_tsne_by_beam(day_features, day_beams, output_dir / "tsne_beam_day.png", "(Day)")
    create_tsne_by_beam(night_features, night_beams, output_dir / "tsne_beam_night.png", "(Night)")

    all_features = np.vstack([day_features, night_features])
    all_beams = np.concatenate([day_beams, night_beams])
    create_tsne_by_beam(all_features, all_beams, output_dir / "tsne_beam_all.png", "(All)")

    print("\n=== Done ===")
    print(f"Outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
