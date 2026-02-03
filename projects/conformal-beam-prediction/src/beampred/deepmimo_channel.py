import numpy as np

try:
    import deepmimo as DeepMIMO
    DEEPMIMO_V4 = True
    HAS_DEEPMIMO = True
except ImportError:
    DEEPMIMO_V4 = False
    try:
        import DeepMIMOv3 as DeepMIMO
        HAS_DEEPMIMO = True
    except ImportError:
        try:
            import DeepMIMO
            HAS_DEEPMIMO = True
        except ImportError:
            HAS_DEEPMIMO = False

from beampred.config import N_ANTENNAS


def _load_v4(scenario, n_samples, scenario_folder=None):
    if scenario_folder:
        DeepMIMO.config.set('scenarios_folder', scenario_folder)
    try:
        DeepMIMO.download(scenario)
    except Exception as e:
        print(f"  Download attempt: {e}")

    from deepmimo import ChannelParameters
    cp = ChannelParameters()
    cp.bs_antenna.shape = [N_ANTENNAS, 1]
    cp.ue_antenna.shape = [1, 1]
    cp.num_paths = 25
    cp.ofdm.subcarriers = 1
    cp.ofdm.bandwidth = 0.001

    dataset = DeepMIMO.load(scenario)
    pair = dataset[-1]
    pair.set_channel_params(cp)
    pair.compute_channels()
    channels_all = pair.channels[:, 0, :, 0]
    norms_all = np.linalg.norm(channels_all, axis=1)
    valid_mask = norms_all > 1e-15
    channels_raw = channels_all[valid_mask]
    print(f"  DeepMIMO v4: {valid_mask.sum()}/{len(valid_mask)} users have valid channels")

    if len(channels_raw) < n_samples:
        print(f"  DeepMIMO v4: only {len(channels_raw)} users, requested {n_samples}")
        n_samples = len(channels_raw)
    channels = channels_raw[:n_samples]

    norms = np.linalg.norm(channels, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    channels = channels / norms * np.sqrt(N_ANTENNAS)

    if hasattr(pair, 'rx_pos') and pair.rx_pos is not None:
        locs = pair.rx_pos[valid_mask][:n_samples]
        bs_loc = pair.tx_pos if hasattr(pair, 'tx_pos') else np.zeros(3)
        distances = np.linalg.norm(locs - bs_loc, axis=1)
    else:
        distances = np.random.uniform(10, 200, n_samples)

    return channels, distances


def _load_v3(scenario, n_samples):
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
    channels_raw = bs["user"]["channel"][:, 0, :, 0].squeeze()

    if len(channels_raw) < n_samples:
        print(f"  DeepMIMO v3: only {len(channels_raw)} users, requested {n_samples}")
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


def load_deepmimo_channels(scenario="boston5g_28", n_samples=70000, scenario_folder=None):
    if not HAS_DEEPMIMO:
        raise ImportError("DeepMIMO not installed. pip install DeepMIMO")

    if DEEPMIMO_V4:
        return _load_v4(scenario, n_samples, scenario_folder)
    return _load_v3(scenario, n_samples)


def try_load_deepmimo(n_samples=70000, scenario=None, scenario_folder=None):
    if scenario:
        scenarios = [scenario]
    else:
        scenarios = ["boston5g_28", "o1_28", "o1_28b", "i3_28"]
    for sc in scenarios:
        try:
            print(f"  Trying DeepMIMO scenario: {sc}")
            channels, distances = load_deepmimo_channels(sc, n_samples, scenario_folder)
            print(f"  Loaded {len(channels)} channels from {sc}")
            return channels, distances, sc
        except Exception as e:
            print(f"  Failed: {e}")
    return None, None, None
