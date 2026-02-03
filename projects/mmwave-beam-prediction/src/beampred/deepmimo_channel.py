import numpy as np

try:
    import deepmimo
    HAS_DEEPMIMO = True
except ImportError:
    HAS_DEEPMIMO = False

from beampred.config import N_ANTENNAS


def load_deepmimo_channels(scenario="boston5g_28", n_samples=70000, scenario_folder=None):
    if not HAS_DEEPMIMO:
        raise ImportError("DeepMIMO not installed. pip install DeepMIMO>=4.0.0b9")

    if scenario_folder:
        deepmimo.config.set('scenarios_folder', scenario_folder)

    dataset = deepmimo.load(scenario, max_paths=10)
    ds = dataset.datasets[0]

    n_ue = min(ds.n_ue, n_samples)
    channels = ds.channel[:n_ue, 0, :, 0]
    aoa_az = ds.aoa_az[:n_ue]
    distances = ds.distance[:n_ue]

    h_array = np.zeros((n_ue, N_ANTENNAS), dtype=complex)
    for i in range(n_ue):
        for p in range(channels.shape[1]):
            if np.isnan(aoa_az[i, p]):
                continue
            theta = np.deg2rad(aoa_az[i, p])
            steering = np.exp(1j * np.pi * np.arange(N_ANTENNAS) * np.sin(theta))
            h_array[i] += channels[i, p] * steering / np.sqrt(N_ANTENNAS)

    valid_mask = np.linalg.norm(h_array, axis=1) > 1e-10
    h_array = h_array[valid_mask]
    distances = distances[valid_mask]

    if len(h_array) < n_samples:
        print(f"  DeepMIMO: only {len(h_array)} valid channels, requested {n_samples}")

    norms = np.linalg.norm(h_array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    h_array = h_array / norms * np.sqrt(N_ANTENNAS)

    return h_array[:n_samples], distances[:n_samples]


def try_load_deepmimo(n_samples=70000, scenario=None, scenario_folder=None):
    if scenario is not None:
        scenarios = [scenario]
    else:
        scenarios = ["boston5g_28", "O1_28"]
    for sc in scenarios:
        try:
            print(f"  Trying DeepMIMO scenario: {sc}")
            channels, distances = load_deepmimo_channels(sc, n_samples, scenario_folder)
            print(f"  Loaded {len(channels)} channels from {sc}")
            return channels, distances, sc
        except Exception as e:
            print(f"  Failed: {e}")
    return None, None, None
