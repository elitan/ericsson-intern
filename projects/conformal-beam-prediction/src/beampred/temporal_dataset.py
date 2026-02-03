"""Sliding-window sequential dataset for temporal beam prediction.

Splits by UE (not time) to avoid data leakage.
Input: (batch, seq_len, n_wide_beams) power vectors.
Target: next narrow beam index.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from beampred.config import (
    SEQ_LEN, BATCH_SIZE, N_WIDE_BEAMS, N_NARROW_BEAMS, CALIBRATION_SAMPLES
)
from beampred.codebook import generate_dft_codebook
from beampred.sionna_channel import load_channels, N_TX as SIONNA_N_TX


class SequentialBeamDataset(Dataset):
    """Sliding window over per-UE time series.

    features: (n_samples, seq_len, n_wide_beams)
    labels: (n_samples,) — beam index at t+1
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def channels_to_wide_beam_powers(channels, n_tx):
    narrow_cb = generate_dft_codebook(n_tx, N_NARROW_BEAMS)
    wide_cb = generate_dft_codebook(n_tx, N_WIDE_BEAMS)

    narrow_gains = np.abs(channels @ narrow_cb.conj().T) ** 2
    labels = np.argmax(narrow_gains, axis=1)

    wide_gains = np.abs(channels @ wide_cb.conj().T) ** 2
    powers_db = 10 * np.log10(np.maximum(wide_gains, 1e-30))

    return powers_db, labels


def build_sequences(powers_db, labels, seq_len=SEQ_LEN):
    """Build sliding window sequences from a single UE's time series.

    Input shape: powers_db (n_timesteps, n_wide), labels (n_timesteps,)
    Output: features (n_windows, seq_len, n_wide), targets (n_windows,)
    Target is the beam at the timestep right after the window.
    """
    n_timesteps = len(labels)
    if n_timesteps <= seq_len:
        return np.empty((0, seq_len, powers_db.shape[1])), np.empty(0, dtype=int)

    n_windows = n_timesteps - seq_len
    idx = np.arange(seq_len)[None, :] + np.arange(n_windows)[:, None]
    features = powers_db[idx]
    targets = labels[seq_len:]

    return features, targets


def prepare_temporal_data(speed_kmh, seed=42, seq_len=SEQ_LEN, train_frac=0.7,
                          cal_frac=0.1, val_frac=0.1):
    """Load Sionna channels, build sequences, split by UE.

    Returns dict with train/cal/val/test features, labels, and metadata.
    """
    channels, distances = load_channels(speed_kmh, seed=seed)
    n_ues, n_timesteps, n_tx = channels.shape

    flat = channels.reshape(n_ues * n_timesteps, n_tx)
    powers_flat, labels_flat = channels_to_wide_beam_powers(flat, n_tx)
    powers_3d = powers_flat.reshape(n_ues, n_timesteps, -1)
    labels_2d = labels_flat.reshape(n_ues, n_timesteps)

    all_features = []
    all_labels = []
    ue_indices = []

    for ue in range(n_ues):
        feat, tgt = build_sequences(powers_3d[ue], labels_2d[ue], seq_len)
        if len(tgt) > 0:
            all_features.append(feat)
            all_labels.append(tgt)
            ue_indices.extend([ue] * len(tgt))

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    ue_indices = np.array(ue_indices)

    rng = np.random.default_rng(seed)
    ue_ids = np.unique(ue_indices)
    rng.shuffle(ue_ids)

    n_train_ues = int(len(ue_ids) * train_frac)
    n_cal_ues = int(len(ue_ids) * cal_frac)
    n_val_ues = int(len(ue_ids) * val_frac)

    train_ues = set(ue_ids[:n_train_ues])
    cal_ues = set(ue_ids[n_train_ues:n_train_ues + n_cal_ues])
    val_ues = set(ue_ids[n_train_ues + n_cal_ues:n_train_ues + n_cal_ues + n_val_ues])
    test_ues = set(ue_ids[n_train_ues + n_cal_ues + n_val_ues:])

    def select(ue_set):
        mask = np.isin(ue_indices, list(ue_set))
        return all_features[mask], all_labels[mask]

    train_feat, train_labels = select(train_ues)
    cal_feat, cal_labels = select(cal_ues)
    val_feat, val_labels = select(val_ues)
    test_feat, test_labels = select(test_ues)

    mean = train_feat.mean(axis=(0, 1))
    std = train_feat.std(axis=(0, 1)) + 1e-8
    train_feat = (train_feat - mean) / std
    cal_feat = (cal_feat - mean) / std
    val_feat = (val_feat - mean) / std
    test_feat = (test_feat - mean) / std

    return {
        "train_feat": train_feat, "train_labels": train_labels,
        "cal_feat": cal_feat, "cal_labels": cal_labels,
        "val_feat": val_feat, "val_labels": val_labels,
        "test_feat": test_feat, "test_labels": test_labels,
        "mean": mean, "std": std,
        "speed_kmh": speed_kmh, "seq_len": seq_len,
        "distances": distances,
    }


def get_temporal_dataloaders(speed_kmh, seed=42, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    data = prepare_temporal_data(speed_kmh, seed=seed, seq_len=seq_len)

    train_ds = SequentialBeamDataset(data["train_feat"], data["train_labels"])
    cal_ds = SequentialBeamDataset(data["cal_feat"], data["cal_labels"])
    val_ds = SequentialBeamDataset(data["val_feat"], data["val_labels"])
    test_ds = SequentialBeamDataset(data["test_feat"], data["test_labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader, "cal": cal_loader,
        "val": val_loader, "test": test_loader,
        "mean": data["mean"], "std": data["std"],
        "test_labels": data["test_labels"],
        "distances": data["distances"],
    }
