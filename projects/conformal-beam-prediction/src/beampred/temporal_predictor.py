"""Temporal beam prediction models: LSTM (primary) and Transformer (secondary).

Input: (batch, seq_len, n_wide_beams) wide-beam power sequences.
Output: (batch, n_narrow_beams) logits for next beam.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from beampred import config
from beampred.config import (
    N_WIDE_BEAMS, N_NARROW_BEAMS, SEQ_LEN, LR,
    EARLY_STOP_PATIENCE, MIXUP_ALPHA
)


class BeamMLP(nn.Module):
    """Spatial-only baseline: uses only last timestep's wide-beam features."""
    def __init__(self, n_input=N_WIDE_BEAMS, n_output=N_NARROW_BEAMS,
                 hidden_size=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_output),
        )

    def forward(self, x):
        return self.net(x[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BeamLSTM(nn.Module):
    def __init__(self, n_input=N_WIDE_BEAMS, n_output=N_NARROW_BEAMS,
                 hidden_size=128, n_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_output),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BeamTemporalTransformer(nn.Module):
    def __init__(self, n_input=N_WIDE_BEAMS, n_output=N_NARROW_BEAMS,
                 seq_len=SEQ_LEN, d_model=64, nhead=2, n_layers=2, dim_ff=128,
                 dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, n_output),
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def temporal_mixup_batch(features, labels, alpha, n_classes, rng):
    if alpha <= 0:
        return features, nn.functional.one_hot(labels, n_classes).float()
    lam = rng.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(features.size(0), device=features.device)
    mixed_features = lam * features + (1 - lam) * features[idx]
    y_a = nn.functional.one_hot(labels, n_classes).float()
    y_b = nn.functional.one_hot(labels[idx], n_classes).float()
    mixed_labels = lam * y_a + (1 - lam) * y_b
    return mixed_features, mixed_labels


def train_temporal_model(train_loader, val_loader, device="cpu", verbose=True,
                         model=None, epochs=None, lr=LR):
    if epochs is None:
        epochs = config.EPOCHS
    if model is None:
        model = BeamLSTM()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    warmup_epochs = 5
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    rng = np.random.default_rng(42)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    t_start = time.time()
    for epoch in range(epochs):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            mixed_feat, mixed_labels = temporal_mixup_batch(
                features, labels, MIXUP_ALPHA, N_NARROW_BEAMS, rng
            )
            logits = model(mixed_feat)
            loss = -torch.sum(mixed_labels * torch.log_softmax(logits, dim=1)) / logits.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(val_total, 1)
        train_acc = correct / max(total, 1)

        epoch_dur = time.time() - t_epoch
        if verbose and (epoch + 1) % 5 == 0:
            remaining = epoch_dur * (epochs - epoch - 1)
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train loss: {train_loss/max(total,1):.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Val acc: {val_acc:.4f} | "
                f"{epoch_dur:.1f}s/ep, ~{remaining:.0f}s left"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
    return model
