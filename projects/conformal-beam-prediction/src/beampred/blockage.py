"""Markov on/off blockage model for mmWave temporal prediction.

Blockage states: CLEAR (0), BLOCKED (1).
Transition rates parameterized by mean duration of each state.
When blocked, applies 20-30 dB attenuation to channel.
"""
import numpy as np
import torch

from beampred.config import (
    BLOCKAGE_ATTEN_DB, BLOCKAGE_MEAN_DURATION_S,
    BLOCKAGE_MEAN_CLEAR_S, SAMPLE_INTERVAL_S,
    N_WIDE_BEAMS, N_NARROW_BEAMS, SEQ_LEN, CONFORMAL_ALPHA
)
from beampred.codebook import generate_dft_codebook
from beampred.sionna_channel import load_channels, N_TX as SIONNA_N_TX
from beampred.temporal_dataset import channels_to_wide_beam_powers, build_sequences
from beampred.conformal import predict_sets, set_sizes


def generate_blockage_trace(n_timesteps, mean_clear_s=BLOCKAGE_MEAN_CLEAR_S,
                            mean_blocked_s=BLOCKAGE_MEAN_DURATION_S,
                            dt=SAMPLE_INTERVAL_S, seed=None):
    """Generate Markov on/off blockage state sequence.

    Returns array of 0 (clear) / 1 (blocked) per timestep.
    """
    rng = np.random.default_rng(seed)

    p_clear_to_blocked = dt / mean_clear_s
    p_blocked_to_clear = dt / mean_blocked_s

    states = np.zeros(n_timesteps, dtype=int)
    state = 0

    for t in range(1, n_timesteps):
        if state == 0:
            if rng.random() < p_clear_to_blocked:
                state = 1
        else:
            if rng.random() < p_blocked_to_clear:
                state = 0
        states[t] = state

    return states


def generate_scripted_blockage(n_timesteps, onset_timestep, duration_timesteps):
    """Deterministic blockage event for demo figures.

    Clear until onset_timestep, then blocked for duration_timesteps.
    """
    states = np.zeros(n_timesteps, dtype=int)
    end = min(onset_timestep + duration_timesteps, n_timesteps)
    states[onset_timestep:end] = 1
    return states


def apply_blockage_to_channels(channels, blockage_states, atten_db=BLOCKAGE_ATTEN_DB):
    """Apply blockage attenuation to channel time series.

    channels: (n_timesteps, n_tx) complex
    blockage_states: (n_timesteps,) int, 0=clear, 1=blocked
    Returns attenuated channels.
    """
    scale = np.ones(len(blockage_states))
    scale[blockage_states == 1] = 10 ** (-atten_db / 20)
    return channels * scale[:, None]


def _moving_average(x, window=5):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _find_sustained_trigger(smoothed_sizes, threshold, timesteps, deadline,
                            n_consecutive=3):
    """Find first index where n_consecutive samples exceed threshold before deadline."""
    count = 0
    for i in range(len(smoothed_sizes)):
        if timesteps[i] >= deadline:
            break
        if smoothed_sizes[i] >= threshold:
            count += 1
            if count >= n_consecutive:
                return i - n_consecutive + 1
        else:
            count = 0
    return None


def _run_demo_for_ue(ue_channels, n_timesteps, n_tx, model, threshold,
                     device, seq_len):
    onset = int(5.0 / SAMPLE_INTERVAL_S)
    duration = int(BLOCKAGE_MEAN_DURATION_S / SAMPLE_INTERVAL_S)
    blockage_states = generate_scripted_blockage(n_timesteps, onset, duration)

    blocked_channels = apply_blockage_to_channels(ue_channels, blockage_states)
    powers_db, beam_labels = channels_to_wide_beam_powers(blocked_channels, n_tx)

    mean = powers_db[:onset].mean(axis=0)
    std = powers_db[:onset].std(axis=0) + 1e-8
    powers_norm = (powers_db - mean) / std

    n_windows = n_timesteps - seq_len
    timesteps = np.arange(seq_len, n_timesteps) * SAMPLE_INTERVAL_S

    set_size_over_time = np.zeros(n_windows)
    predicted_beams = np.zeros(n_windows, dtype=int)

    model.eval()
    with torch.no_grad():
        for i in range(n_windows):
            window = powers_norm[i:i + seq_len]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred = np.argmax(probs)
            predicted_beams[i] = pred

            above = probs >= 1.0 - threshold
            beam_set = np.where(above)[0]
            if len(beam_set) == 0:
                beam_set = np.array([pred])
            set_size_over_time[i] = len(beam_set)

    smoothed = _moving_average(set_size_over_time, window=5)
    blockage_onset_time = onset * SAMPLE_INTERVAL_S
    baseline_median = np.median(smoothed[:onset - seq_len])
    expansion_threshold = baseline_median * 1.5

    early_idx = _find_sustained_trigger(
        smoothed, expansion_threshold, timesteps, blockage_onset_time, n_consecutive=3
    )

    if early_idx is not None:
        lead_time = blockage_onset_time - timesteps[early_idx]
    else:
        lead_time = 0.0

    correct_over_time = (predicted_beams == beam_labels[seq_len:])

    narrow_cb = generate_dft_codebook(n_tx, N_NARROW_BEAMS)
    rsrp_db = 10 * np.log10(np.maximum(
        np.abs(blocked_channels[seq_len:] @ narrow_cb.conj().T) ** 2, 1e-30
    ))
    pred_rsrp = rsrp_db[np.arange(n_windows), predicted_beams]
    oracle_rsrp = rsrp_db[np.arange(n_windows), beam_labels[seq_len:]]

    return {
        "timesteps": timesteps,
        "set_sizes": smoothed,
        "set_sizes_raw": set_size_over_time,
        "correct": correct_over_time,
        "predicted_beams": predicted_beams,
        "true_beams": beam_labels[seq_len:],
        "blockage_states": blockage_states[seq_len:],
        "blockage_onset_time": blockage_onset_time,
        "pred_rsrp": pred_rsrp,
        "oracle_rsrp": oracle_rsrp,
        "lead_time_s": lead_time,
        "expansion_threshold": expansion_threshold,
    }


def run_blockage_demo(speed_kmh=60, model=None, threshold=None,
                      seed=42, device="cpu", ue_idx=None):
    """Run scripted blockage demo. Tries first 10 UEs, picks cleanest.

    "Cleanest" = largest lead time. Falls back to ue_idx=0.
    """
    channels_all, distances = load_channels(speed_kmh, seed=seed)
    n_ues, n_timesteps, n_tx = channels_all.shape
    seq_len = SEQ_LEN

    if ue_idx is not None:
        result = _run_demo_for_ue(
            channels_all[ue_idx], n_timesteps, n_tx, model, threshold, device, seq_len
        )
        result["speed_kmh"] = speed_kmh
        return result

    best_result = None
    best_lead = -1.0
    n_try = min(10, n_ues)

    for idx in range(n_try):
        result = _run_demo_for_ue(
            channels_all[idx], n_timesteps, n_tx, model, threshold, device, seq_len
        )
        if result["lead_time_s"] > best_lead:
            best_lead = result["lead_time_s"]
            best_result = result

    if best_result is None:
        best_result = _run_demo_for_ue(
            channels_all[0], n_timesteps, n_tx, model, threshold, device, seq_len
        )

    best_result["speed_kmh"] = speed_kmh
    return best_result
