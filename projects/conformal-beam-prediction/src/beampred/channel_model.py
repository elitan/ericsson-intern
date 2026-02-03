import numpy as np
from beampred.config import (
    N_ANTENNAS, N_CLUSTERS, N_RAYS_PER_CLUSTER,
    ANGULAR_SPREAD_DEG, D_MIN, D_MAX, FREQ
)


def generate_channels(n_samples, seed=42):
    rng = np.random.default_rng(seed)

    distances = rng.uniform(D_MIN, D_MAX, n_samples)
    los_probs = np.clip(18.0 / distances, 0.0, 1.0)
    los_flags = rng.random(n_samples) < los_probs

    max_clusters = N_CLUSTERS + 2
    sigma_rad = np.deg2rad(ANGULAR_SPREAD_DEG)

    n_arr = np.arange(N_ANTENNAS)
    d_over_lam = 0.5

    channels = np.zeros((n_samples, N_ANTENNAS), dtype=complex)

    for c in range(max_clusters):
        active_los = (c < 1)
        active_nlos = (c < max_clusters)

        active = np.where(los_flags, active_los, active_nlos)
        if not np.any(active):
            continue

        idx = np.where(active)[0]
        n_active = len(idx)

        cluster_aoa = rng.uniform(-np.pi / 2, np.pi / 2, n_active)
        cluster_power = np.exp(-c * 3.0)

        for r in range(N_RAYS_PER_CLUSTER):
            ray_offsets = rng.laplace(0, sigma_rad / np.sqrt(2), n_active)
            ray_aoa = np.clip(cluster_aoa + ray_offsets, -np.pi / 2, np.pi / 2)

            alpha_re = rng.standard_normal(n_active)
            alpha_im = rng.standard_normal(n_active)
            alpha = (alpha_re + 1j * alpha_im) / np.sqrt(2)
            alpha *= np.sqrt(cluster_power / N_RAYS_PER_CLUSTER)

            phases = 2 * np.pi * d_over_lam * np.outer(np.sin(ray_aoa), n_arr)
            steering = np.exp(1j * phases) / np.sqrt(N_ANTENNAS)

            channels[idx] += alpha[:, None] * steering

    norms = np.linalg.norm(channels, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    channels = channels / norms * np.sqrt(N_ANTENNAS)

    pl_db = 32.4 + 21.0 * np.log10(distances) + 20.0 * np.log10(FREQ / 1e9)
    scale = 10 ** (-pl_db / 20)
    channels = channels * scale[:, None]

    return channels, distances, los_flags


def generate_channel(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    channels, distances, los_flags = generate_channels(1, seed=rng.integers(0, 2**31))
    return channels[0], distances[0], los_flags[0]
