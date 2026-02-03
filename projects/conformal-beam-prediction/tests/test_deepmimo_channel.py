import pytest
import numpy as np

try:
    import DeepMIMO
    HAS_DEEPMIMO = True
except ImportError:
    HAS_DEEPMIMO = False

from beampred.deepmimo_channel import load_deepmimo_channels, try_load_deepmimo


skip_no_deepmimo = pytest.mark.skipif(not HAS_DEEPMIMO, reason="DeepMIMO not installed")


@skip_no_deepmimo
class TestLoadDeepMIMOChannels:
    def test_o1_28_shape(self):
        channels, distances = load_deepmimo_channels("O1_28", n_samples=100)
        assert channels.shape[1] == 64
        assert np.iscomplexobj(channels)
        assert len(distances) == len(channels)

    def test_i3_28_shape(self):
        channels, distances = load_deepmimo_channels("I3_28", n_samples=100)
        assert channels.shape[1] == 64
        assert np.iscomplexobj(channels)

    def test_distances_positive_finite(self):
        channels, distances = load_deepmimo_channels("O1_28", n_samples=100)
        assert np.all(distances > 0)
        assert np.all(np.isfinite(distances))


@skip_no_deepmimo
class TestTryLoadDeepMIMO:
    def test_specific_scenario(self):
        channels, distances, sc = try_load_deepmimo(n_samples=100, scenario="O1_28")
        assert channels is not None
        assert sc == "O1_28"
        assert channels.shape[1] == 64

    def test_fallback(self):
        channels, distances, sc = try_load_deepmimo(n_samples=100)
        if channels is not None:
            assert channels.shape[1] == 64
            assert sc is not None
