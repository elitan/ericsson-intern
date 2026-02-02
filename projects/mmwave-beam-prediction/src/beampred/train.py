import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from beampred.config import LR, EPOCHS, EARLY_STOP_PATIENCE, LABEL_SMOOTHING, N_NARROW_BEAMS
from beampred.beam_predictor import BeamPredictor

WARMUP_EPOCHS = 5
MIXUP_ALPHA = 0.4


def mixup_batch(features, labels, alpha, n_classes, rng):
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


def train_model(train_loader, val_loader, device="cpu", verbose=True, model=None):
    if model is None:
        model = BeamPredictor()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    rng = np.random.default_rng(42)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    t_start = time.time()
    for epoch in range(EPOCHS):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            mixed_feat, mixed_labels = mixup_batch(features, labels, MIXUP_ALPHA, N_NARROW_BEAMS, rng)
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

        val_acc = val_correct / val_total
        train_acc = correct / total

        epoch_dur = time.time() - t_epoch
        elapsed = time.time() - t_start
        if verbose and (epoch + 1) % 5 == 0:
            remaining = epoch_dur * (EPOCHS - epoch - 1)
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train loss: {train_loss/total:.4f} | "
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

    model.load_state_dict(best_state)
    if verbose:
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
    return model
