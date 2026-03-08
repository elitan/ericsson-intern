"""Evaluation metrics for beam prediction."""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as float
    """
    _, top_k_preds = torch.topk(logits, k, dim=1)
    correct = (top_k_preds == targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k_values: list[int] = [1, 3, 5],
) -> dict[str, float]:
    """
    Evaluate model on dataloader.

    Returns:
        Dictionary with top-k accuracies and loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            gps = batch["gps"].to(device)
            targets = batch["beam_index"].to(device)

            logits = model(images, gps)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    results = {
        "loss": total_loss / len(dataloader.dataset),
    }

    for k in top_k_values:
        results[f"top_{k}_accuracy"] = top_k_accuracy(all_logits, all_targets, k)

    return results


def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 64,
) -> np.ndarray:
    """Compute confusion matrix."""
    preds = torch.argmax(logits, dim=1).numpy()
    targets = targets.numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(preds, targets):
        cm[target, pred] += 1

    return cm


def per_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 64,
) -> np.ndarray:
    """Compute per-class accuracy."""
    cm = compute_confusion_matrix(logits, targets, num_classes)
    per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
    return per_class


class MetricsTracker:
    """Track training and validation metrics over time."""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "top_1_accuracy": [],
            "top_3_accuracy": [],
            "top_5_accuracy": [],
        }

    def update(self, metrics: dict[str, float], prefix: str = ""):
        """Add metrics to history."""
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)

    def get_best(self, metric: str = "top_1_accuracy") -> tuple[int, float]:
        """Get epoch and value of best metric."""
        values = self.history.get(metric, [])
        if not values:
            return -1, 0.0
        best_idx = np.argmax(values)
        return best_idx, values[best_idx]

    def summary(self) -> str:
        """Get summary string of current metrics."""
        lines = []
        for key, values in self.history.items():
            if values:
                lines.append(f"{key}: {values[-1]:.4f}")
        return " | ".join(lines)
