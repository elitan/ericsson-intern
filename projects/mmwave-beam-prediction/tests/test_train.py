import torch
import numpy as np
from beampred.train import mixup_batch


class TestMixupBatch:
    def test_no_mixup(self):
        features = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])
        mixed_f, mixed_l = mixup_batch(features, labels, alpha=0, n_classes=4, rng=np.random.default_rng(0))
        torch.testing.assert_close(mixed_f, features)
        assert mixed_l.shape == (4, 4)

    def test_with_mixup_shape(self):
        features = torch.randn(8, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        rng = np.random.default_rng(42)
        mixed_f, mixed_l = mixup_batch(features, labels, alpha=0.4, n_classes=2, rng=rng)
        assert mixed_f.shape == features.shape
        assert mixed_l.shape == (8, 2)

    def test_labels_sum_to_one(self):
        features = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 3])
        rng = np.random.default_rng(0)
        _, mixed_l = mixup_batch(features, labels, alpha=0.4, n_classes=4, rng=rng)
        sums = mixed_l.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(4))
