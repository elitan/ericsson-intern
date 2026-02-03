"""Offline 3GPP TR 38.901 UMa channel generation via Sionna.

TensorFlow stays isolated here. Main pipeline never imports TF.
Run as: uv run python -m beampred.sionna_channel --speed 30 --n-ues 500 --duration 10
Output: data/sionna/uma_{speed}kmh_seed{seed}.npz
"""
import argparse
import os
import numpy as np

from beampred.config import DATA_DIR


SIONNA_DIR = os.path.join(DATA_DIR, "sionna")

CARRIER_FREQ = 30e9
N_TX = 32
N_RX = 1
SAMPLE_INTERVAL_S = 0.02
SUBCARRIER_SPACING = 30e3
FFT_SIZE = 128
N_OFDM_SYMBOLS = 14
SCENARIO = "uma"
TX_HEIGHT = 25.0
RX_HEIGHT = 1.5
MIN_DIST = 35.0
MAX_DIST = 500.0


def generate_sionna_channels(speed_kmh, n_ues, duration_s, seed=42):
    """Generate time-series UMa channels using Sionna ray tracing.

    Returns (n_ues, n_timesteps, n_tx) complex channel matrix.
    Falls back to synthetic Doppler channels if Sionna unavailable.
    """
    n_timesteps = int(duration_s / SAMPLE_INTERVAL_S)

    try:
        return _generate_with_sionna(speed_kmh, n_ues, n_timesteps, seed)
    except (ImportError, Exception) as e:
        print(f"  Sionna unavailable ({e}), using synthetic Doppler channels")
        return _generate_synthetic_doppler(speed_kmh, n_ues, n_timesteps, seed)


def _generate_with_sionna(speed_kmh, n_ues, n_timesteps, seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.random.set_seed(seed)

    import sionna
    from sionna.channel.tr38901 import UMa
    from sionna.channel import gen_single_sector_topology

    channel_model = UMa(
        carrier_frequency=CARRIER_FREQ,
        o2i_model="low",
        ut_array=sionna.channel.tr38901.Antenna(polarization="single", polarization_type="V",
                                                  antenna_pattern="omni", carrier_frequency=CARRIER_FREQ),
        bs_array=sionna.channel.tr38901.AntennaArray(num_rows=1, num_cols=N_TX,
                                                      polarization="single", polarization_type="V",
                                                      antenna_pattern="38.901", carrier_frequency=CARRIER_FREQ),
        direction="downlink",
        enable_pathloss=True,
        enable_shadow_fading=True,
    )

    speed_ms = speed_kmh / 3.6
    all_channels = np.zeros((n_ues, n_timesteps, N_TX), dtype=complex)
    distances = np.zeros(n_ues)

    batch_size = min(n_ues, 100)
    n_batches = (n_ues + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_ues)
        bs = end - start

        topology = gen_single_sector_topology(
            batch_size=bs,
            num_ut=1,
            scenario="uma",
            min_ut_velocity=speed_ms,
            max_ut_velocity=speed_ms,
        )
        channel_model.set_topology(*topology)

        for t in range(n_timesteps):
            h = channel_model(
                num_time_samples=1,
                sampling_frequency=SUBCARRIER_SPACING * FFT_SIZE,
            )
            h_np = h[0].numpy()
            all_channels[start:end, t, :] = h_np[:, 0, 0, :, 0, 0]

        distances[start:end] = np.linalg.norm(
            topology[1].numpy()[:, 0, :2], axis=1
        )

    return all_channels, distances


def _generate_synthetic_doppler(speed_kmh, n_ues, n_timesteps, seed):
    """UE-trajectory channel model with moving scatterers geometry.

    Each UE starts near origin, moves at configured speed in random direction.
    3-5 scatterer clusters placed 50-200m away. AoAs recomputed each timestep
    from atan2(scatterer - ue_pos). Cluster power decays with distance.
    Produces real beam transitions over time.
    """
    rng = np.random.default_rng(seed)
    c = 3e8
    wavelength = c / CARRIER_FREQ
    speed_ms = speed_kmh / 3.6
    f_doppler = speed_ms / wavelength

    n_clusters = rng.integers(3, 6, size=n_ues)
    max_clusters = 5
    n_rays = 10
    n_arr = np.arange(N_TX)

    channels = np.zeros((n_ues, n_timesteps, N_TX), dtype=complex)
    distances = rng.uniform(MIN_DIST, MAX_DIST, n_ues)

    ue_start_x = rng.uniform(-10, 10, n_ues)
    ue_start_y = rng.uniform(-10, 10, n_ues)
    travel_dir = rng.uniform(0, 2 * np.pi, n_ues)
    ue_vx = speed_ms * np.cos(travel_dir)
    ue_vy = speed_ms * np.sin(travel_dir)

    scat_r = rng.uniform(50, 200, (n_ues, max_clusters))
    scat_angle = rng.uniform(0, 2 * np.pi, (n_ues, max_clusters))
    scat_x = ue_start_x[:, None] + scat_r * np.cos(scat_angle)
    scat_y = ue_start_y[:, None] + scat_r * np.sin(scat_angle)

    cluster_base_power = np.exp(-np.arange(max_clusters) * 1.5)[None, :]
    cluster_mask = np.arange(max_clusters)[None, :] < n_clusters[:, None]

    ray_aoa_offsets = rng.laplace(0, np.deg2rad(5) / np.sqrt(2),
                                  (n_ues, max_clusters, n_rays))
    ray_phases = rng.uniform(0, 2 * np.pi, (n_ues, max_clusters, n_rays))
    ray_amplitudes = (rng.standard_normal((n_ues, max_clusters, n_rays))
                      + 1j * rng.standard_normal((n_ues, max_clusters, n_rays))) / np.sqrt(2)

    for t in range(n_timesteps):
        time_s = t * SAMPLE_INTERVAL_S
        ue_x = ue_start_x + ue_vx * time_s
        ue_y = ue_start_y + ue_vy * time_s

        dx = scat_x - ue_x[:, None]
        dy = scat_y - ue_y[:, None]
        dist_to_scat = np.sqrt(dx**2 + dy**2)
        cluster_aoas = np.arctan2(dy, dx)
        cluster_aoas = np.clip(cluster_aoas, -np.pi / 2, np.pi / 2)

        dist_decay = np.exp(-dist_to_scat / 150.0)
        cluster_power = cluster_base_power * dist_decay * cluster_mask

        h = np.zeros((n_ues, N_TX), dtype=complex)

        for c_idx in range(max_clusters):
            power = np.sqrt(np.maximum(cluster_power[:, c_idx], 1e-30) / n_rays)

            for r in range(n_rays):
                aoa = cluster_aoas[:, c_idx] + ray_aoa_offsets[:, c_idx, r]
                aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

                doppler_shift = f_doppler * np.cos(aoa - travel_dir)
                phase_shift = 2 * np.pi * doppler_shift * time_s + ray_phases[:, c_idx, r]

                spatial_phase = 2 * np.pi * 0.5 * np.outer(np.sin(aoa), n_arr)
                steering = np.exp(1j * spatial_phase) / np.sqrt(N_TX)

                alpha = ray_amplitudes[:, c_idx, r] * power * np.exp(1j * phase_shift)
                h += alpha[:, None] * steering

        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-30)
        h = h / norms * np.sqrt(N_TX)

        pl_db = 32.4 + 21.0 * np.log10(distances) + 20.0 * np.log10(CARRIER_FREQ / 1e9)
        scale = 10 ** (-pl_db / 20)
        h = h * scale[:, None]

        channels[:, t, :] = h

    return channels, distances


def save_channels(channels, distances, speed_kmh, seed):
    os.makedirs(SIONNA_DIR, exist_ok=True)
    path = os.path.join(SIONNA_DIR, f"uma_{speed_kmh}kmh_seed{seed}.npz")
    np.savez_compressed(
        path,
        channels=channels,
        distances=distances,
        speed_kmh=speed_kmh,
        carrier_freq=CARRIER_FREQ,
        n_tx=N_TX,
        sample_interval_s=SAMPLE_INTERVAL_S,
    )
    print(f"  Saved {path}: shape={channels.shape}, {os.path.getsize(path)/1e6:.1f} MB")
    return path


def load_channels(speed_kmh, seed=42):
    path = os.path.join(SIONNA_DIR, f"uma_{speed_kmh}kmh_seed{seed}.npz")
    data = np.load(path)
    return data["channels"], data["distances"]


def main():
    parser = argparse.ArgumentParser(description="Generate 3GPP UMa channels")
    parser.add_argument("--speed", type=int, default=30, help="UE speed in km/h")
    parser.add_argument("--n-ues", type=int, default=500)
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-speeds", action="store_true", help="Generate for 3,30,60,120 km/h")
    args = parser.parse_args()

    speeds = [3, 30, 60, 120] if args.all_speeds else [args.speed]

    for speed in speeds:
        print(f"\nGenerating UMa channels at {speed} km/h...")
        channels, distances = generate_sionna_channels(
            speed, args.n_ues, args.duration, args.seed
        )
        print(f"  Shape: {channels.shape} (n_ues, n_timesteps, n_tx)")
        save_channels(channels, distances, speed, args.seed)


if __name__ == "__main__":
    main()
