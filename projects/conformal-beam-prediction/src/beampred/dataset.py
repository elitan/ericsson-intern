import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from beampred.config import (
    N_TRAIN, N_VAL, N_TEST, N_WIDE_BEAMS, N_NARROW_BEAMS, BATCH_SIZE, DATA_DIR, SEED,
    CALIBRATION_SAMPLES
)
from beampred.codebook import get_narrow_codebook, get_wide_codebook, generate_dft_codebook
from beampred.channel_model import generate_channels
from beampred import deepmimo_channel
from beampred.sionna_channel import load_channels as load_sionna_channels, N_TX as SIONNA_N_TX


def compute_features_and_labels(channels, n_antennas=None):
    if n_antennas is None:
        n_antennas = channels.shape[1]

    if n_antennas == N_WIDE_BEAMS * 4:
        narrow_cb = get_narrow_codebook()
        wide_cb = get_wide_codebook()
    else:
        n_narrow = N_NARROW_BEAMS
        n_wide = N_WIDE_BEAMS
        narrow_cb = generate_dft_codebook(n_antennas, n_narrow)
        wide_cb = generate_dft_codebook(n_antennas, n_wide)

    narrow_gains = np.abs(channels @ narrow_cb.conj().T) ** 2
    labels = np.argmax(narrow_gains, axis=1)

    wide_gains = np.abs(channels @ wide_cb.conj().T) ** 2
    wide_powers_db = 10 * np.log10(np.maximum(wide_gains, 1e-30))

    return wide_powers_db, labels


class BeamDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def standardize(train_feat, *other_feats):
    mean = train_feat.mean(axis=0)
    std = train_feat.std(axis=0) + 1e-8
    result = [(train_feat - mean) / std]
    for feat in other_feats:
        result.append((feat - mean) / std)
    return tuple(result) + (mean, std)


def generate_or_load_data(seed=SEED, use_cache=True, source="synthetic", scenario="O1_28"):
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = f"_{scenario}" if source == "deepmimo" else ""
    cache_path = os.path.join(DATA_DIR, f"dataset_{source}{suffix}_seed{seed}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return (
            data["train_feat"], data["train_labels"], data["train_distances"],
            data["cal_feat"], data["cal_labels"], data["cal_distances"],
            data["val_feat"], data["val_labels"], data["val_distances"],
            data["test_feat"], data["test_labels"], data["test_distances"],
            data["test_channels"],
        )

    n_total = N_TRAIN + CALIBRATION_SAMPLES + N_VAL + N_TEST

    if source == "deepmimo":
        scenario_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deepmimo_scenarios")
        channels, distances, scenario = deepmimo_channel.try_load_deepmimo(n_total, scenario=scenario, scenario_folder=scenario_folder)
        if channels is None:
            print("  DeepMIMO failed, falling back to synthetic channels")
            source = "synthetic"

    if source == "sionna":
        from beampred.config import SPEEDS_KMH
        speed = SPEEDS_KMH[0]
        raw_channels, distances = load_sionna_channels(speed, seed=seed)
        n_ues, n_timesteps, n_tx = raw_channels.shape
        channels = raw_channels.reshape(n_ues * n_timesteps, n_tx)
        distances = np.repeat(distances, n_timesteps)
        if len(channels) < n_total:
            print(f"  Warning: Sionna data has {len(channels)} < {n_total} needed, cycling")
            reps = (n_total // len(channels)) + 1
            channels = np.tile(channels, (reps, 1))[:n_total]
            distances = np.tile(distances, reps)[:n_total]
        else:
            channels = channels[:n_total]
            distances = distances[:n_total]

    if source == "synthetic":
        channels, distances, _ = generate_channels(n_total, seed=seed)

    features, labels = compute_features_and_labels(channels)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(features))
    features, labels, distances, channels = features[idx], labels[idx], distances[idx], channels[idx]

    i1 = N_TRAIN
    i2 = i1 + CALIBRATION_SAMPLES
    i3 = i2 + N_VAL
    i4 = i3 + N_TEST

    result = (
        features[:i1], labels[:i1], distances[:i1],
        features[i1:i2], labels[i1:i2], distances[i1:i2],
        features[i2:i3], labels[i2:i3], distances[i2:i3],
        features[i3:i4], labels[i3:i4], distances[i3:i4],
        channels[i3:i4],
    )

    np.savez_compressed(
        cache_path,
        train_feat=result[0], train_labels=result[1], train_distances=result[2],
        cal_feat=result[3], cal_labels=result[4], cal_distances=result[5],
        val_feat=result[6], val_labels=result[7], val_distances=result[8],
        test_feat=result[9], test_labels=result[10], test_distances=result[11],
        test_channels=result[12],
    )

    return result


def get_dataloaders(seed=SEED, use_cache=True, source="synthetic", scenario="O1_28"):
    data = generate_or_load_data(seed, use_cache, source, scenario=scenario)
    train_feat, train_labels, train_dist = data[0], data[1], data[2]
    cal_feat, cal_labels, cal_dist = data[3], data[4], data[5]
    val_feat, val_labels, val_dist = data[6], data[7], data[8]
    test_feat, test_labels, test_dist = data[9], data[10], data[11]
    test_channels = data[12]

    train_feat, cal_feat, val_feat, test_feat, mean, std = standardize(
        train_feat, cal_feat, val_feat, test_feat
    )

    train_ds = BeamDataset(train_feat, train_labels)
    cal_ds = BeamDataset(cal_feat, cal_labels)
    val_ds = BeamDataset(val_feat, val_labels)
    test_ds = BeamDataset(test_feat, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    cal_loader = DataLoader(cal_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, cal_loader, val_loader, test_loader,
            test_channels, test_dist, cal_dist, test_labels, mean, std)
