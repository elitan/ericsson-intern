import numpy as np
import pytest
from beampred.sionna_channel import (
    _generate_synthetic_doppler, save_channels, load_channels, SIONNA_DIR,
    N_TX, CARRIER_FREQ, SAMPLE_INTERVAL_S
)
from beampred.codebook import generate_dft_codebook


def test_synthetic_doppler_shape():
    channels, distances = _generate_synthetic_doppler(30, 10, 50, seed=42)
    assert channels.shape == (10, 50, N_TX)
    assert distances.shape == (10,)
    assert channels.dtype == complex


def test_synthetic_doppler_temporal_correlation():
    channels, _ = _generate_synthetic_doppler(3, 50, 100, seed=42)
    norms = np.linalg.norm(channels, axis=2, keepdims=True) + 1e-30
    ch_norm = channels / norms
    per_ue_corr = np.abs(np.sum(ch_norm[:, :-1].conj() * ch_norm[:, 1:], axis=2))
    mean_corr = np.mean(per_ue_corr)
    assert mean_corr > 0.3, f"Low temporal correlation at 3 km/h: {mean_corr}"


def test_synthetic_doppler_speed_dependence():
    ch_slow, _ = _generate_synthetic_doppler(3, 20, 50, seed=42)
    ch_fast, _ = _generate_synthetic_doppler(120, 20, 50, seed=42)

    def avg_corr(ch):
        c = np.sum(ch[:, :-1].conj() * ch[:, 1:], axis=2)
        n = np.linalg.norm(ch[:, :-1], axis=2) * np.linalg.norm(ch[:, 1:], axis=2) + 1e-30
        return np.mean(np.abs(c / n))

    assert avg_corr(ch_slow) > avg_corr(ch_fast), "Slow should be more correlated than fast"


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr("beampred.sionna_channel.SIONNA_DIR", str(tmp_path))
    channels, distances = _generate_synthetic_doppler(30, 5, 20, seed=42)
    save_channels(channels, distances, 30, 42)

    monkeypatch.setattr("beampred.sionna_channel.SIONNA_DIR", str(tmp_path))
    loaded_ch, loaded_dist = load_channels(30, seed=42)
    np.testing.assert_array_almost_equal(channels, loaded_ch)
    np.testing.assert_array_almost_equal(distances, loaded_dist)


def test_distances_in_range():
    _, distances = _generate_synthetic_doppler(30, 100, 10, seed=42)
    assert np.all(distances >= 35.0)
    assert np.all(distances <= 500.0)


def test_beam_transitions_exist():
    """At 120 km/h over 10s, best narrow beam should change for most UEs."""
    n_ues, n_timesteps = 50, 500
    channels, _ = _generate_synthetic_doppler(120, n_ues, n_timesteps, seed=42)
    cb = generate_dft_codebook(N_TX, 64)
    gains = np.abs(channels.reshape(-1, N_TX) @ cb.conj().T) ** 2
    beams = np.argmax(gains, axis=1).reshape(n_ues, n_timesteps)

    ues_with_transitions = 0
    for ue in range(n_ues):
        if len(np.unique(beams[ue])) > 1:
            ues_with_transitions += 1

    frac = ues_with_transitions / n_ues
    assert frac > 0.5, f"Only {frac:.0%} of UEs had beam transitions at 120 km/h"
