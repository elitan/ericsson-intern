"""Beam prediction models for multimodal input."""

import torch
import torch.nn as nn
from torchvision import models


class GPSEncoder(nn.Module):
    """Encode GPS coordinates to feature vector."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ImageEncoder(nn.Module):
    """Encode images using pretrained ResNet."""

    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class BeamPredictor(nn.Module):
    """
    Multimodal beam predictor combining image and GPS features.

    Architecture:
    - ResNet18 for image encoding
    - MLP for GPS encoding
    - Fusion via concatenation + MLP
    - Output: 64-way beam classification
    """

    def __init__(
        self,
        num_beams: int = 64,
        image_feat_dim: int = 512,
        gps_feat_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(output_dim=image_feat_dim, pretrained=pretrained)
        self.gps_encoder = GPSEncoder(output_dim=gps_feat_dim)

        combined_dim = image_feat_dim + gps_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim, num_beams)

    def forward(
        self,
        image: torch.Tensor,
        gps: torch.Tensor,
    ) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        gps_feat = self.gps_encoder(gps)

        combined = torch.cat([image_feat, gps_feat], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)

        return logits


class ImageOnlyPredictor(nn.Module):
    """Image-only beam predictor (ablation baseline)."""

    def __init__(
        self,
        num_beams: int = 64,
        image_feat_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim=image_feat_dim, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_beams),
        )

    def forward(self, image: torch.Tensor, gps: torch.Tensor = None) -> torch.Tensor:
        feat = self.image_encoder(image)
        return self.classifier(feat)


class GPSOnlyPredictor(nn.Module):
    """GPS-only beam predictor (position baseline)."""

    def __init__(
        self,
        num_beams: int = 64,
        gps_feat_dim: int = 128,
    ):
        super().__init__()
        self.gps_encoder = GPSEncoder(output_dim=gps_feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(gps_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_beams),
        )

    def forward(self, image: torch.Tensor = None, gps: torch.Tensor = None) -> torch.Tensor:
        feat = self.gps_encoder(gps)
        return self.classifier(feat)


def create_model(
    model_type: str = "multimodal",
    num_beams: int = 64,
    pretrained: bool = True,
) -> nn.Module:
    """Factory function for creating beam prediction models."""
    if model_type == "multimodal":
        return BeamPredictor(num_beams=num_beams, pretrained=pretrained)
    elif model_type == "image_only":
        return ImageOnlyPredictor(num_beams=num_beams, pretrained=pretrained)
    elif model_type == "gps_only":
        return GPSOnlyPredictor(num_beams=num_beams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
