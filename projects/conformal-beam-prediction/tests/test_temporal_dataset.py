import numpy as np
import pytest
from beampred.temporal_dataset import (
    build_sequences, channels_to_wide_beam_powers, SequentialBeamDataset
)
from beampred.config import N_WIDE_BEAMS, N_NARROW_BEAMS


def test_build_sequences_shape():
    n_timesteps = 50
    n_wide = N_WIDE_BEAMS
    powers = np.random.randn(n_timesteps, n_wide)
    labels = np.random.randint(0, N_NARROW_BEAMS, n_timesteps)

    seq_len = 10
    features, targets = build_sequences(powers, labels, seq_len)

    assert features.shape == (n_timesteps - seq_len, seq_len, n_wide)
    assert targets.shape == (n_timesteps - seq_len,)


def test_build_sequences_target_alignment():
    n_timesteps = 20
    powers = np.random.randn(n_timesteps, 4)
    labels = np.arange(n_timesteps)

    features, targets = build_sequences(powers, labels, seq_len=5)
    assert targets[0] == 5
    assert targets[1] == 6
    assert targets[-1] == n_timesteps - 1


def test_build_sequences_too_short():
    powers = np.random.randn(5, 4)
    labels = np.arange(5)
    features, targets = build_sequences(powers, labels, seq_len=5)
    assert len(targets) == 0


def test_channels_to_wide_beam_powers_shape():
    n_tx = 32
    channels = (np.random.randn(100, n_tx) + 1j * np.random.randn(100, n_tx)) / np.sqrt(n_tx)
    powers, labels = channels_to_wide_beam_powers(channels, n_tx)
    assert powers.shape == (100, N_WIDE_BEAMS)
    assert labels.shape == (100,)
    assert np.all(labels >= 0) and np.all(labels < N_NARROW_BEAMS)


def test_sequential_dataset():
    features = np.random.randn(100, 10, 16).astype(np.float32)
    labels = np.random.randint(0, 64, 100)
    ds = SequentialBeamDataset(features, labels)
    assert len(ds) == 100
    x, y = ds[0]
    assert x.shape == (10, 16)
    assert y.shape == ()
