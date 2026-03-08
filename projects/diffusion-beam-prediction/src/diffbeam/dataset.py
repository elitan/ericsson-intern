"""DeepSense 6G dataset loader for multimodal beam prediction."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepSenseDataset(Dataset):
    """
    DeepSense 6G dataset for scenarios 31-34 (multimodal beam prediction).

    Structure:
    scenarioXX/
      scenarioXX.csv
      unit1/
        camera_data/image_XXX.jpg
        GPS_data/gps_location.txt
        mmWave_data/mmWave_power_XXX.txt
      unit2/
        GPS_data/GPS_location_XXX.txt
    """

    def __init__(
        self,
        data_dir: str | Path,
        scenarios: list[int] = [31, 32, 33, 34],
        split: str = "train",
        train_ratio: float = 0.8,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.scenarios = scenarios
        self.split = split
        self.seed = seed

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        self.samples = []
        self.gps_stats = None
        self._load_samples(train_ratio)

    def _load_samples(self, train_ratio: float):
        """Load sample metadata from all scenarios."""
        all_samples = []

        for scenario in self.scenarios:
            scenario_dir = self.data_dir / f"scenario{scenario}"
            if not scenario_dir.exists():
                print(f"Warning: scenario {scenario} not found, skipping")
                continue

            samples = self._load_scenario(scenario_dir, scenario)
            all_samples.extend(samples)

        if not all_samples:
            print(f"Warning: No samples found in {self.data_dir}")
            return

        self._compute_gps_stats(all_samples)

        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(indices) * train_ratio)

        if self.split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.samples = [all_samples[i] for i in selected]
        print(f"Loaded {len(self.samples)} samples for {self.split} split from {len(self.scenarios)} scenarios")

    def _load_scenario(self, scenario_dir: Path, scenario: int) -> list[dict]:
        """Load samples from a scenario using the CSV index."""
        samples = []

        csv_path = scenario_dir / f"scenario{scenario}.csv"
        if not csv_path.exists():
            csv_files = list(scenario_dir.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
            else:
                print(f"Warning: No CSV found in {scenario_dir}")
                return samples

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_rel = row["unit1_rgb"]
            img_path = scenario_dir / img_rel
            if not img_path.exists():
                continue

            gps_rel = row["unit2_loc"]
            gps_path = scenario_dir / gps_rel
            if gps_path.exists():
                try:
                    gps_data = np.loadtxt(gps_path)
                    if gps_data.shape == (2,):
                        gps = gps_data.astype(np.float32)
                    else:
                        gps = np.array([33.42, -111.93], dtype=np.float32)
                except:
                    gps = np.array([33.42, -111.93], dtype=np.float32)
            else:
                gps = np.array([33.42, -111.93], dtype=np.float32)

            beam_index = int(row["unit1_beam"]) - 1

            samples.append({
                "image_path": str(img_path),
                "gps": gps,
                "beam_index": beam_index,
                "scenario": scenario,
            })

        print(f"  Scenario {scenario}: {len(samples)} valid samples")
        return samples

    def _compute_gps_stats(self, samples):
        """Compute GPS mean/std for normalization."""
        gps_array = np.array([s["gps"] for s in samples])
        self.gps_stats = {
            "mean": gps_array.mean(axis=0),
            "std": gps_array.std(axis=0) + 1e-6,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gps = torch.tensor(sample["gps"], dtype=torch.float32)
        if self.gps_stats:
            gps = (gps - torch.tensor(self.gps_stats["mean"])) / torch.tensor(self.gps_stats["std"])

        beam_index = torch.tensor(sample["beam_index"], dtype=torch.long)

        return {
            "image": image,
            "gps": gps,
            "beam_index": beam_index,
            "scenario": sample["scenario"],
        }


class SyntheticDeepSenseDataset(Dataset):
    """Synthetic dataset for testing when real data unavailable."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_beams: int = 64,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_beams = num_beams
        self.image_size = image_size

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.gps_coords = np.random.randn(num_samples, 2).astype(np.float32)

        gps_to_beam = (
            (self.gps_coords[:, 0] > 0).astype(int) * 32 +
            (self.gps_coords[:, 1] > 0).astype(int) * 16 +
            np.random.randint(0, 16, num_samples)
        )
        self.beam_indices = gps_to_beam % num_beams

        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        image = torch.rand(3, self.image_size, self.image_size)

        beam = self.beam_indices[idx]
        color_factor = beam / self.num_beams
        image[0] *= color_factor
        image[2] *= (1 - color_factor)

        image = self.transform(image)

        gps = torch.tensor(self.gps_coords[idx], dtype=torch.float32)
        beam_index = torch.tensor(self.beam_indices[idx], dtype=torch.long)

        return {
            "image": image,
            "gps": gps,
            "beam_index": beam_index,
            "scenario": 0,
        }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_synthetic: bool = False,
    scenarios: list[int] = [31, 32, 33, 34],
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""
    if use_synthetic:
        train_dataset = SyntheticDeepSenseDataset(num_samples=5000)
        val_dataset = SyntheticDeepSenseDataset(num_samples=1000, seed=43)
    else:
        train_dataset = DeepSenseDataset(data_dir, scenarios=scenarios, split="train")
        val_dataset = DeepSenseDataset(data_dir, scenarios=scenarios, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
