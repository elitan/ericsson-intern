import numpy as np

try:
    import DeepMIMO
    HAS_DEEPMIMO = True
except ImportError:
    HAS_DEEPMIMO = False

from beampred.config import N_ANTENNAS


def load_deepmimo_channels(scenario="O1_28", n_samples=70000):
    if not HAS_DEEPMIMO:
        raise ImportError("DeepMIMO not installed. pip install DeepMIMO")

    params = DeepMIMO.default_params()
    params["scenario"] = scenario
    params["active_BS"] = np.array([1])
    params["user_row_first"] = 1
    params["user_row_last"] = 502
    params["bs_antenna"]["shape"] = np.array([N_ANTENNAS, 1, 1])
    params["ue_antenna"]["shape"] = np.array([1, 1, 1])
    params["num_paths"] = 10
    params["OFDM"]["subcarriers"] = 1
    params["OFDM"]["bandwidth"] = 0.1

    dataset = DeepMIMO.generate_data(params)

    bs = dataset[0]
    n_users = bs["user"]["channel"].shape[0]
    channels_raw = bs["user"]["channel"][:, 0, :, 0].squeeze()

    if len(channels_raw) < n_samples:
        print(f"  DeepMIMO: only {len(channels_raw)} users available, requested {n_samples}")
        n_samples = len(channels_raw)

    channels = channels_raw[:n_samples]

    norms = np.linalg.norm(channels, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    channels = channels / norms * np.sqrt(N_ANTENNAS)

    if "user" in bs and "location" in bs["user"]:
        locs = bs["user"]["location"][:n_samples]
        bs_loc = bs["location"]
        distances = np.linalg.norm(locs - bs_loc, axis=1)
    else:
        distances = np.random.uniform(10, 200, n_samples)

    return channels, distances


def try_load_deepmimo(n_samples=70000):
    scenarios = ["O1_28", "O1_28B", "I3_28", "Boston5G_28"]
    for sc in scenarios:
        try:
            print(f"  Trying DeepMIMO scenario: {sc}")
            channels, distances = load_deepmimo_channels(sc, n_samples)
            print(f"  Loaded {len(channels)} channels from {sc}")
            return channels, distances, sc
        except Exception as e:
            print(f"  Failed: {e}")
    return None, None, None
