import numpy as np
from beampred.error_analysis import beam_error_distances, error_distance_histogram, cost_weighted_score, run_error_analysis
from beampred.channel_model import generate_channels
from beampred.exhaustive_search import exhaustive_search_batch


class TestBeamErrorDistances:
    def test_zero_when_correct(self):
        pred = np.array([0, 5, 10])
        true = np.array([0, 5, 10])
        np.testing.assert_array_equal(beam_error_distances(pred, true), [0, 0, 0])

    def test_absolute_distance(self):
        pred = np.array([3, 10])
        true = np.array([5, 7])
        np.testing.assert_array_equal(beam_error_distances(pred, true), [2, 3])


class TestErrorDistanceHistogram:
    def test_all_correct(self):
        pred = np.array([0, 1, 2])
        true = np.array([0, 1, 2])
        hist = error_distance_histogram(pred, true)
        assert np.all(hist == 0)

    def test_some_wrong(self):
        pred = np.array([0, 3])
        true = np.array([0, 5])
        hist = error_distance_histogram(pred, true)
        assert hist[1] == 1  # distance=2 at index 1


class TestCostWeightedScore:
    def test_perfect_prediction(self):
        channels, _, _ = generate_channels(50, seed=0)
        beams, _, _ = exhaustive_search_batch(channels)
        score = cost_weighted_score(beams, beams, channels)
        np.testing.assert_allclose(score, 1.0, atol=1e-6)


class TestRunErrorAnalysis:
    def test_returns_expected_keys(self):
        channels, _, _ = generate_channels(50, seed=0)
        beams, _, _ = exhaustive_search_batch(channels)
        result = run_error_analysis(beams, beams, channels)
        assert "error_histogram" in result
        assert "gain_loss_per_distance" in result
        assert "cost_weighted_score" in result
