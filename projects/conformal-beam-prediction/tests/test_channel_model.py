import numpy as np
from beampred.channel_model import generate_channels, generate_channel
from beampred.config import N_ANTENNAS, D_MIN, D_MAX


class TestGenerateChannels:
    def test_shape(self):
        channels, distances, los = generate_channels(50, seed=0)
        assert channels.shape == (50, N_ANTENNAS)
        assert distances.shape == (50,)
        assert los.shape == (50,)

    def test_complex(self):
        channels, _, _ = generate_channels(10, seed=0)
        assert np.iscomplexobj(channels)

    def test_distance_range(self):
        _, distances, _ = generate_channels(500, seed=0)
        assert distances.min() >= D_MIN
        assert distances.max() <= D_MAX

    def test_deterministic(self):
        c1, _, _ = generate_channels(20, seed=7)
        c2, _, _ = generate_channels(20, seed=7)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds(self):
        c1, _, _ = generate_channels(20, seed=1)
        c2, _, _ = generate_channels(20, seed=2)
        assert not np.allclose(c1, c2)

    def test_los_flags_bool(self):
        _, _, los = generate_channels(100, seed=0)
        assert los.dtype == bool

    def test_nonzero_channels(self):
        channels, _, _ = generate_channels(50, seed=0)
        norms = np.linalg.norm(channels, axis=1)
        assert np.all(norms > 0)


class TestGenerateChannel:
    def test_single_output(self):
        ch, d, los = generate_channel()
        assert ch.shape == (N_ANTENNAS,)
        assert isinstance(d, (float, np.floating))
        assert isinstance(los, (bool, np.bool_))
