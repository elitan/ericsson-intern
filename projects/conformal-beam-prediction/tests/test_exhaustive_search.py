import numpy as np
from beampred.exhaustive_search import exhaustive_search, exhaustive_search_batch
from beampred.channel_model import generate_channels
from beampred.codebook import get_narrow_codebook


class TestExhaustiveSearch:
    def setup_method(self):
        self.channels, _, _ = generate_channels(20, seed=0)
        self.codebook = get_narrow_codebook()

    def test_returns_valid_beam(self):
        best, gain, gains = exhaustive_search(self.channels[0], self.codebook)
        assert 0 <= best < 64
        assert gain > 0
        assert gains.shape == (64,)

    def test_best_is_max(self):
        best, gain, gains = exhaustive_search(self.channels[0], self.codebook)
        assert gain == gains.max()
        assert best == np.argmax(gains)


class TestExhaustiveSearchBatch:
    def setup_method(self):
        self.channels, _, _ = generate_channels(50, seed=0)
        self.codebook = get_narrow_codebook()

    def test_shape(self):
        beams, gains, all_gains = exhaustive_search_batch(self.channels, self.codebook)
        assert beams.shape == (50,)
        assert gains.shape == (50,)
        assert all_gains.shape == (50, 64)

    def test_matches_single(self):
        beams_batch, _, _ = exhaustive_search_batch(self.channels, self.codebook)
        for i in range(5):
            beam_single, _, _ = exhaustive_search(self.channels[i], self.codebook)
            assert beams_batch[i] == beam_single
