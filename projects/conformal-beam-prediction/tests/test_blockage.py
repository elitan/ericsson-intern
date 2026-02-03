import numpy as np
import pytest
from beampred.blockage import (
    generate_blockage_trace, generate_scripted_blockage,
    apply_blockage_to_channels
)


def test_blockage_trace_shape():
    trace = generate_blockage_trace(500, seed=42)
    assert trace.shape == (500,)
    assert set(np.unique(trace)).issubset({0, 1})


def test_blockage_trace_starts_clear():
    trace = generate_blockage_trace(100, seed=42)
    assert trace[0] == 0


def test_blockage_trace_has_transitions():
    trace = generate_blockage_trace(10000, mean_clear_s=0.5, mean_blocked_s=0.5,
                                    dt=0.02, seed=42)
    assert np.sum(np.diff(trace) != 0) > 10, "Expected transitions in long trace"


def test_scripted_blockage():
    trace = generate_scripted_blockage(100, onset_timestep=30, duration_timesteps=20)
    assert np.all(trace[:30] == 0)
    assert np.all(trace[30:50] == 1)
    assert np.all(trace[50:] == 0)


def test_scripted_blockage_overflow():
    trace = generate_scripted_blockage(50, onset_timestep=40, duration_timesteps=20)
    assert np.all(trace[:40] == 0)
    assert np.all(trace[40:50] == 1)


def test_apply_blockage():
    n_tx = 32
    channels = np.ones((10, n_tx), dtype=complex)
    states = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    result = apply_blockage_to_channels(channels, states, atten_db=20)

    expected_scale = 10 ** (-20 / 20)
    np.testing.assert_almost_equal(np.abs(result[0, 0]), 1.0)
    np.testing.assert_almost_equal(np.abs(result[3, 0]), expected_scale, decimal=5)
    np.testing.assert_almost_equal(np.abs(result[7, 0]), 1.0)
