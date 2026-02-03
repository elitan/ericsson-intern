"""
Handover predictor model.

Simple MLP that predicts target cell from RSRP measurements.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class HandoverMLP(nn.Module):
    def __init__(self, n_cells: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = n_cells + n_cells + 1  # rsrp + serving_cell_onehot + speed
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_cells),
        )
        self.n_cells = n_cells

    def forward(self, rsrp, serving_cell, speed):
        serving_onehot = torch.zeros(rsrp.shape[0], self.n_cells, device=rsrp.device)
        serving_onehot.scatter_(1, serving_cell.unsqueeze(1), 1)
        x = torch.cat([rsrp, serving_onehot, speed.unsqueeze(1)], dim=1)
        return self.net(x)


def prepare_features(data: dict) -> tuple:
    """Convert data dict to tensors."""
    rsrp = torch.tensor(data["rsrp"], dtype=torch.float32)
    serving = torch.tensor(data["serving_cell"], dtype=torch.long)
    optimal = torch.tensor(data["optimal_cell"], dtype=torch.long)
    speed = torch.tensor(data["ue_speed"], dtype=torch.float32)
    speed = (speed - speed.mean()) / (speed.std() + 1e-8)
    rsrp = (rsrp - rsrp.mean()) / (rsrp.std() + 1e-8)
    return rsrp, serving, optimal, speed


def train_predictor(
    data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> HandoverMLP:
    """Train handover predictor."""
    rsrp, serving, optimal, speed = prepare_features(data)
    n_cells = data["n_cells"]

    train_ds = TensorDataset(
        rsrp[train_idx], serving[train_idx], optimal[train_idx], speed[train_idx]
    )
    val_ds = TensorDataset(
        rsrp[val_idx], serving[val_idx], optimal[val_idx], speed[val_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = HandoverMLP(n_cells).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for rsrp_b, serving_b, optimal_b, speed_b in train_loader:
            rsrp_b = rsrp_b.to(device)
            serving_b = serving_b.to(device)
            optimal_b = optimal_b.to(device)
            speed_b = speed_b.to(device)

            optimizer.zero_grad()
            logits = model(rsrp_b, serving_b, speed_b)
            loss = criterion(logits, optimal_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rsrp_b, serving_b, optimal_b, speed_b in val_loader:
                rsrp_b = rsrp_b.to(device)
                serving_b = serving_b.to(device)
                optimal_b = optimal_b.to(device)
                speed_b = speed_b.to(device)

                logits = model(rsrp_b, serving_b, speed_b)
                preds = logits.argmax(dim=1)
                correct += (preds == optimal_b).sum().item()
                total += len(optimal_b)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")

    return model


def get_softmax_scores(
    model: HandoverMLP,
    data: dict,
    idx: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Get softmax probabilities for samples."""
    rsrp, serving, optimal, speed = prepare_features(data)

    model.eval()
    with torch.no_grad():
        rsrp_b = rsrp[idx].to(device)
        serving_b = serving[idx].to(device)
        speed_b = speed[idx].to(device)
        logits = model(rsrp_b, serving_b, speed_b)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()
