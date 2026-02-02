import torch
from beampred.complexity import count_params, estimate_flops_fc
from beampred.beam_predictor import BeamPredictor


class TestCountParams:
    def test_matches_model_method(self):
        model = BeamPredictor(n_input=16, n_output=64, hidden_dims=[32, 32])
        assert count_params(model) == model.count_parameters()


class TestEstimateFlops:
    def test_positive(self):
        model = BeamPredictor(n_input=16, n_output=64, hidden_dims=[32])
        flops = estimate_flops_fc(model)
        assert flops > 0

    def test_bigger_model_more_flops(self):
        small = BeamPredictor(n_input=16, n_output=64, hidden_dims=[16])
        big = BeamPredictor(n_input=16, n_output=64, hidden_dims=[256, 256])
        assert estimate_flops_fc(big) > estimate_flops_fc(small)
