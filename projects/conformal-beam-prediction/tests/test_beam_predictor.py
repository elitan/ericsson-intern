import torch
import pytest
from beampred.beam_predictor import BeamPredictor, ResBlock, ResNetMLP, BeamTransformer


class TestBeamPredictor:
    def test_output_shape(self):
        model = BeamPredictor(n_input=16, n_output=64)
        x = torch.randn(8, 16)
        out = model(x)
        assert out.shape == (8, 64)

    def test_count_parameters(self):
        model = BeamPredictor(n_input=16, n_output=64, hidden_dims=[32, 32])
        assert model.count_parameters() > 0

    def test_different_hidden_dims(self):
        model = BeamPredictor(n_input=4, n_output=8, hidden_dims=[16])
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 8)


class TestResBlock:
    def test_residual_shape(self):
        block = ResBlock(32)
        x = torch.randn(4, 32)
        out = block(x)
        assert out.shape == (4, 32)


class TestResNetMLP:
    def test_output_shape(self):
        model = ResNetMLP(n_input=16, n_output=64, hidden=32, n_blocks=2)
        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape == (4, 64)

    def test_count_parameters(self):
        model = ResNetMLP(n_input=16, n_output=64, hidden=32, n_blocks=2)
        assert model.count_parameters() > 0


class TestBeamTransformer:
    def test_output_shape(self):
        model = BeamTransformer(n_input=16, n_output=64, d_model=32, nhead=2, n_layers=1)
        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape == (4, 64)

    def test_count_parameters(self):
        model = BeamTransformer(n_input=16, n_output=64)
        assert model.count_parameters() > 0
