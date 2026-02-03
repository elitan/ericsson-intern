import numpy as np
from beampred.baselines import hierarchical_search
from beampred.channel_model import generate_channels
from beampred.config import N_NARROW_BEAMS, N_WIDE_BEAMS


class TestHierarchicalSearch:
    def setup_method(self):
        self.channels, _, _ = generate_channels(50, seed=0)

    def test_valid_beams(self):
        beams, overhead = hierarchical_search(self.channels)
        assert beams.shape == (50,)
        assert np.all(beams >= 0)
        assert np.all(beams < N_NARROW_BEAMS)

    def test_overhead(self):
        _, overhead = hierarchical_search(self.channels)
        ratio = N_NARROW_BEAMS // N_WIDE_BEAMS
        assert overhead == N_WIDE_BEAMS + ratio
