import numpy as np
import torch
from beampred.evaluate import top_k_accuracy, spectral_efficiency
from beampred.codebook import get_narrow_codebook


class TestTopKAccuracy:
    def test_top1_perfect(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = torch.tensor([0, 1])
        assert top_k_accuracy(logits, labels, 1) == 1.0

    def test_top1_wrong(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1])
        assert top_k_accuracy(logits, labels, 1) == 0.5

    def test_top3_relaxes(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
        labels = torch.tensor([1])
        assert top_k_accuracy(logits, labels, 1) == 0.0
        assert top_k_accuracy(logits, labels, 3) == 1.0


class TestSpectralEfficiency:
    def test_positive(self):
        cb = get_narrow_codebook()
        channel = cb[0]  # perfectly aligned
        se = spectral_efficiency(channel, 0, cb, 10.0)
        assert se > 0

    def test_higher_snr_higher_se(self):
        cb = get_narrow_codebook()
        channel = cb[5]
        se_low = spectral_efficiency(channel, 5, cb, 1.0)
        se_high = spectral_efficiency(channel, 5, cb, 100.0)
        assert se_high > se_low
