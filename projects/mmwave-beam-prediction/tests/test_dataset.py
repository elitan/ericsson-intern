import numpy as np
import torch
from beampred.dataset import compute_features_and_labels, BeamDataset, standardize
from beampred.channel_model import generate_channels


class TestComputeFeaturesAndLabels:
    def setup_method(self):
        self.channels, _, _ = generate_channels(100, seed=0)

    def test_shapes(self):
        features, labels = compute_features_and_labels(self.channels)
        assert features.shape == (100, 16)
        assert labels.shape == (100,)

    def test_labels_valid_range(self):
        _, labels = compute_features_and_labels(self.channels)
        assert labels.min() >= 0
        assert labels.max() < 64

    def test_features_finite(self):
        features, _ = compute_features_and_labels(self.channels)
        assert np.all(np.isfinite(features))


class TestBeamDataset:
    def test_len(self):
        ds = BeamDataset(np.random.randn(50, 16), np.random.randint(0, 64, 50))
        assert len(ds) == 50

    def test_getitem(self):
        ds = BeamDataset(np.random.randn(10, 16), np.random.randint(0, 64, 10))
        feat, label = ds[0]
        assert feat.shape == (16,)
        assert isinstance(label, torch.Tensor)


class TestStandardize:
    def test_zero_mean_unit_var(self):
        train = np.random.randn(200, 4)
        result = standardize(train)
        normed_train = result[0]
        np.testing.assert_allclose(normed_train.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(normed_train.std(axis=0), 1, atol=0.1)

    def test_multiple_sets(self):
        train = np.random.randn(200, 4)
        val = np.random.randn(50, 4)
        test = np.random.randn(50, 4)
        result = standardize(train, val, test)
        assert len(result) == 5  # 3 datasets + mean + std
