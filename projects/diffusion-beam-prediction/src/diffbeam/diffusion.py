"""Diffusion-based data augmentation for beam prediction."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from torchvision import transforms
from PIL import Image
import numpy as np


class BeamConditionedDiffusion:
    """
    Diffusion model for generating augmented images conditioned on beam class.

    Uses HuggingFace diffusers with class conditioning to generate
    synthetic training images for specific beam indices.
    """

    def __init__(
        self,
        image_size: int = 64,
        num_beams: int = 64,
        num_train_timesteps: int = 1000,
        device: str = "auto",
    ):
        self.image_size = image_size
        self.num_beams = num_beams

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            num_class_embeds=num_beams,
        ).to(self.device)

        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize([-1], [2]),
        ])

    def train_step(
        self,
        images: torch.Tensor,
        beam_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step."""
        images = images.to(self.device)
        beam_indices = beam_indices.to(self.device)

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (images.shape[0],), device=self.device
        ).long()

        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        noise_pred = self.model(
            noisy_images,
            timesteps,
            class_labels=beam_indices,
        ).sample

        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def generate(
        self,
        beam_indices: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate images for given beam indices."""
        batch_size = beam_indices.shape[0]
        beam_indices = beam_indices.to(self.device)

        images = torch.randn(
            batch_size, 3, self.image_size, self.image_size,
            device=self.device
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            noise_pred = self.model(
                images,
                t,
                class_labels=beam_indices,
            ).sample

            images = self.scheduler.step(noise_pred, t, images).prev_sample

        images = self.inverse_transform(images)
        images = torch.clamp(images, 0, 1)

        return images

    def save(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "image_size": self.image_size,
            "num_beams": self.num_beams,
        }, path / "diffusion_model.pt")

    def load(self, path: str | Path):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path / "diffusion_model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


class DiffusionAugmenter:
    """
    Augmenter that generates synthetic samples using trained diffusion model.
    """

    def __init__(
        self,
        diffusion_model: BeamConditionedDiffusion,
        target_size: int = 224,
    ):
        self.diffusion = diffusion_model
        self.resize = transforms.Resize((target_size, target_size))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @torch.no_grad()
    def augment_batch(
        self,
        beam_indices: torch.Tensor,
        gps_coords: torch.Tensor,
        gps_noise_std: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate augmented images and perturbed GPS for given beam indices.

        Returns:
            Tuple of (augmented_images, augmented_gps)
        """
        generated = self.diffusion.generate(beam_indices)

        generated = self.resize(generated)
        generated = self.normalize(generated)

        perturbed_gps = gps_coords + torch.randn_like(gps_coords) * gps_noise_std

        return generated, perturbed_gps


class TraditionalAugmenter:
    """Traditional augmentation for comparison baseline."""

    def __init__(self, p: float = 0.5):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 4:
            return torch.stack([self._augment_single(img) for img in image])
        return self._augment_single(image)

    def _augment_single(self, image: torch.Tensor) -> torch.Tensor:
        image_pil = transforms.ToPILImage()(image)
        augmented = self.transform(image_pil)
        return transforms.ToTensor()(augmented)
