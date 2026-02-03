"""
Synthetic handover dataset generator.

Simulates:
- Grid of base stations (gNBs)
- UEs moving along linear/random trajectories
- RSRP measurements from serving + neighboring cells
- Ground truth: optimal cell based on strongest RSRP
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    n_gnb_x: int = 3
    n_gnb_y: int = 3
    cell_radius: float = 200.0  # meters
    frequency_ghz: float = 3.5
    tx_power_dbm: float = 46.0
    noise_std_db: float = 4.0  # shadow fading std


@dataclass
class MobilityConfig:
    n_trajectories: int = 1000
    trajectory_length: int = 100  # time steps
    speed_range: tuple = (1.0, 30.0)  # m/s (walking to driving)
    dt: float = 0.1  # seconds per step


def path_loss_db(distance_m: np.ndarray, freq_ghz: float) -> np.ndarray:
    """3GPP Urban Micro path loss model (simplified)."""
    d = np.maximum(distance_m, 1.0)
    pl = 32.4 + 20 * np.log10(freq_ghz) + 30 * np.log10(d)
    return pl


def generate_gnb_positions(config: NetworkConfig) -> np.ndarray:
    """Generate grid of gNB positions."""
    spacing = config.cell_radius * np.sqrt(3)
    x = np.arange(config.n_gnb_x) * spacing
    y = np.arange(config.n_gnb_y) * spacing
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return positions


def generate_trajectory(
    gnb_positions: np.ndarray,
    mobility_config: MobilityConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate single UE trajectory."""
    x_min, y_min = gnb_positions.min(axis=0) - 100
    x_max, y_max = gnb_positions.max(axis=0) + 100

    start = rng.uniform([x_min, y_min], [x_max, y_max])
    speed = rng.uniform(*mobility_config.speed_range)
    angle = rng.uniform(0, 2 * np.pi)
    velocity = speed * np.array([np.cos(angle), np.sin(angle)])

    positions = np.zeros((mobility_config.trajectory_length, 2))
    positions[0] = start

    for t in range(1, mobility_config.trajectory_length):
        new_pos = positions[t - 1] + velocity * mobility_config.dt
        if new_pos[0] < x_min or new_pos[0] > x_max:
            velocity[0] *= -1
            new_pos = positions[t - 1] + velocity * mobility_config.dt
        if new_pos[1] < y_min or new_pos[1] > y_max:
            velocity[1] *= -1
            new_pos = positions[t - 1] + velocity * mobility_config.dt
        positions[t] = new_pos

    return positions


def compute_rsrp(
    ue_positions: np.ndarray,
    gnb_positions: np.ndarray,
    config: NetworkConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute RSRP from each UE position to each gNB.

    Returns: (n_positions, n_gnb) array of RSRP values in dBm
    """
    n_pos = ue_positions.shape[0]
    n_gnb = gnb_positions.shape[0]

    distances = np.linalg.norm(
        ue_positions[:, None, :] - gnb_positions[None, :, :], axis=2
    )

    pl = path_loss_db(distances, config.frequency_ghz)
    shadow = rng.normal(0, config.noise_std_db, size=(n_pos, n_gnb))
    rsrp = config.tx_power_dbm - pl + shadow

    return rsrp


def generate_dataset(
    network_config: NetworkConfig = None,
    mobility_config: MobilityConfig = None,
    seed: int = 42,
    prediction_horizon: int = 5,
    measurement_noise_db: float = 3.0,
) -> dict:
    """
    Generate synthetic handover dataset.

    Args:
        prediction_horizon: predict optimal cell this many steps ahead
        measurement_noise_db: additional noise on observed RSRP

    Returns dict with:
        - rsrp: (n_samples, n_gnb) noisy RSRP measurements
        - serving_cell: (n_samples,) current serving cell
        - optimal_cell: (n_samples,) ground truth optimal cell (future)
        - ue_speed: (n_samples,) UE speed
        - trajectory_id: (n_samples,) which trajectory
        - time_step: (n_samples,) time within trajectory
    """
    if network_config is None:
        network_config = NetworkConfig()
    if mobility_config is None:
        mobility_config = MobilityConfig()

    rng = np.random.default_rng(seed)
    gnb_positions = generate_gnb_positions(network_config)
    n_gnb = len(gnb_positions)

    all_rsrp = []
    all_serving = []
    all_optimal = []
    all_speed = []
    all_traj_id = []
    all_time = []

    for traj_id in range(mobility_config.n_trajectories):
        traj = generate_trajectory(gnb_positions, mobility_config, rng)
        rsrp_true = compute_rsrp(traj, gnb_positions, network_config, rng)

        rsrp_noisy = rsrp_true + rng.normal(0, measurement_noise_db, rsrp_true.shape)

        speed = np.linalg.norm(np.diff(traj, axis=0), axis=1) / mobility_config.dt
        speed = np.concatenate([[speed[0]], speed])

        optimal_current = rsrp_true.argmax(axis=1)

        optimal_future = np.zeros(len(traj), dtype=int)
        for t in range(len(traj)):
            future_t = min(t + prediction_horizon, len(traj) - 1)
            optimal_future[t] = rsrp_true[future_t].argmax()

        serving = np.zeros(len(traj), dtype=int)
        serving[0] = optimal_current[0]
        for t in range(1, len(traj)):
            current_rsrp = rsrp_true[t, serving[t - 1]]
            best_rsrp = rsrp_true[t].max()
            if best_rsrp - current_rsrp > 3.0:
                serving[t] = optimal_current[t]
            else:
                serving[t] = serving[t - 1]

        all_rsrp.append(rsrp_noisy)
        all_serving.append(serving)
        all_optimal.append(optimal_future)
        all_speed.append(speed)
        all_traj_id.append(np.full(len(traj), traj_id))
        all_time.append(np.arange(len(traj)))

    return {
        "rsrp": np.concatenate(all_rsrp),
        "serving_cell": np.concatenate(all_serving),
        "optimal_cell": np.concatenate(all_optimal),
        "ue_speed": np.concatenate(all_speed),
        "trajectory_id": np.concatenate(all_traj_id),
        "time_step": np.concatenate(all_time),
        "n_cells": n_gnb,
        "gnb_positions": gnb_positions,
    }


if __name__ == "__main__":
    data = generate_dataset(prediction_horizon=10, measurement_noise_db=4.0)
    print(f"Generated {len(data['rsrp'])} samples")
    print(f"Number of cells: {data['n_cells']}")
    print(f"RSRP shape: {data['rsrp'].shape}")
    print(f"Speed range: {data['ue_speed'].min():.1f} - {data['ue_speed'].max():.1f} m/s")

    n_handovers = np.sum(np.diff(data["serving_cell"]) != 0)
    print(f"Total handovers: {n_handovers}")
