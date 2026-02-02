import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from beampred.conformal import calibrate, predict_sets, calibrate_beam_aware, predict_sets_beam_aware, set_sizes, coverage
from beampred.beam_predictor import BeamPredictor


def _make_loader(n=200, n_classes=8):
    features = torch.randn(n, 4)
    labels = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(features, labels), batch_size=64)


def _make_model():
    return BeamPredictor(n_input=4, n_output=8, hidden_dims=[16])


class TestCalibrate:
    def test_threshold_positive(self):
        model = _make_model()
        loader = _make_loader()
        threshold, scores = calibrate(model, loader, alpha=0.1)
        assert threshold > 0
        assert len(scores) == 200

    def test_lower_alpha_higher_threshold(self):
        model = _make_model()
        loader = _make_loader()
        t1, _ = calibrate(model, loader, alpha=0.5)
        t2, _ = calibrate(model, loader, alpha=0.05)
        assert t2 >= t1


class TestPredictSets:
    def test_returns_nonempty_sets(self):
        model = _make_model()
        loader = _make_loader()
        threshold, _ = calibrate(model, loader, alpha=0.1)
        sets = predict_sets(model, loader, threshold)
        assert all(len(s) >= 1 for s in sets)

    def test_count(self):
        model = _make_model()
        loader = _make_loader(n=100)
        threshold, _ = calibrate(model, loader, alpha=0.1)
        sets = predict_sets(model, loader, threshold)
        assert len(sets) == 100


class TestBeamAware:
    def test_calibrate_returns_threshold(self):
        model = _make_model()
        loader = _make_loader()
        threshold, scores = calibrate_beam_aware(model, loader, alpha=0.1)
        assert threshold > 0

    def test_predict_sets_nonempty(self):
        model = _make_model()
        loader = _make_loader()
        threshold, _ = calibrate_beam_aware(model, loader, alpha=0.1)
        sets = predict_sets_beam_aware(model, loader, threshold)
        assert all(len(s) >= 1 for s in sets)


class TestSetSizes:
    def test_basic(self):
        sets = [np.array([1, 2, 3]), np.array([5]), np.array([0, 1])]
        sizes = set_sizes(sets)
        np.testing.assert_array_equal(sizes, [3, 1, 2])


class TestCoverage:
    def test_perfect_coverage(self):
        sets = [np.array([0, 1, 2]), np.array([3, 4])]
        labels = [1, 4]
        assert coverage(sets, labels) == 1.0

    def test_zero_coverage(self):
        sets = [np.array([0]), np.array([1])]
        labels = [5, 5]
        assert coverage(sets, labels) == 0.0

    def test_partial(self):
        sets = [np.array([0]), np.array([1])]
        labels = [0, 5]
        assert coverage(sets, labels) == 0.5
