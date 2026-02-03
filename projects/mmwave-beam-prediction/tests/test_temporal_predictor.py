import torch
import pytest
from beampred.temporal_predictor import BeamLSTM, BeamTemporalTransformer
from beampred.config import N_WIDE_BEAMS, N_NARROW_BEAMS, SEQ_LEN


@pytest.fixture
def batch():
    return torch.randn(8, SEQ_LEN, N_WIDE_BEAMS)


def test_lstm_output_shape(batch):
    model = BeamLSTM()
    out = model(batch)
    assert out.shape == (8, N_NARROW_BEAMS)


def test_transformer_output_shape(batch):
    model = BeamTemporalTransformer()
    out = model(batch)
    assert out.shape == (8, N_NARROW_BEAMS)


def test_lstm_different_seq_len():
    model = BeamLSTM()
    x = torch.randn(4, 5, N_WIDE_BEAMS)
    out = model(x)
    assert out.shape == (4, N_NARROW_BEAMS)


def test_transformer_count_parameters():
    model = BeamTemporalTransformer()
    n = model.count_parameters()
    assert n > 0
    assert n < 1_000_000


def test_lstm_count_parameters():
    model = BeamLSTM()
    n = model.count_parameters()
    assert n > 0
    assert n < 500_000


def test_lstm_gradients(batch):
    model = BeamLSTM()
    labels = torch.randint(0, N_NARROW_BEAMS, (8,))
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
