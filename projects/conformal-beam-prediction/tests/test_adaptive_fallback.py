import numpy as np
from beampred.adaptive_fallback import adaptive_beam_management, sweep_thresholds
from beampred.channel_model import generate_channels
from beampred.exhaustive_search import exhaustive_search_batch


class TestAdaptiveBeamManagement:
    def setup_method(self):
        self.channels, _, _ = generate_channels(50, seed=0)
        self.true_beams, _, _ = exhaustive_search_batch(self.channels)
        self.prediction_sets = [np.array([b]) for b in self.true_beams]

    def test_perfect_sets_give_perfect_accuracy(self):
        r = adaptive_beam_management(self.prediction_sets, self.true_beams, self.channels, confidence_threshold=3)
        assert r["accuracy"] == 1.0
        assert r["ml_fraction"] == 1.0

    def test_large_sets_fallback(self):
        big_sets = [np.arange(64) for _ in range(50)]
        r = adaptive_beam_management(big_sets, self.true_beams, self.channels, confidence_threshold=3)
        assert r["ml_fraction"] == 0.0
        assert r["accuracy"] == 1.0  # exhaustive always finds optimal

    def test_returns_expected_keys(self):
        r = adaptive_beam_management(self.prediction_sets, self.true_beams, self.channels)
        for key in ["selected_beams", "overheads", "accuracy", "ml_fraction", "avg_overhead", "used_ml"]:
            assert key in r


class TestSweepThresholds:
    def test_returns_list(self):
        channels, _, _ = generate_channels(30, seed=0)
        beams, _, _ = exhaustive_search_batch(channels)
        sets = [np.array([b]) for b in beams]
        results = sweep_thresholds(sets, beams, channels, thresholds=[1, 2, 3])
        assert len(results) == 3
        assert all("threshold" in r for r in results)
