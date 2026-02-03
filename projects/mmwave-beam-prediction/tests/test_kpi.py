import numpy as np
import pytest
from beampred.kpi import (
    l1_rsrp_gap, overhead_slots, effective_throughput,
    compute_beam_rsrp, compute_all_kpis
)
from beampred.config import N_NARROW_BEAMS, N_WIDE_BEAMS


def test_l1_rsrp_gap_perfect():
    n = 100
    beams = np.arange(n) % N_NARROW_BEAMS
    gains = np.random.randn(n, N_NARROW_BEAMS) * 10
    result = l1_rsrp_gap(beams, beams, gains)
    np.testing.assert_array_equal(result["gap_db"], np.zeros(n))
    assert result["mean"] == 0.0


def test_l1_rsrp_gap_nonzero():
    n = 50
    true_beams = np.zeros(n, dtype=int)
    pred_beams = np.ones(n, dtype=int)
    gains = np.zeros((n, N_NARROW_BEAMS))
    gains[:, 0] = 10.0
    gains[:, 1] = 7.0
    result = l1_rsrp_gap(pred_beams, true_beams, gains)
    np.testing.assert_almost_equal(result["gap_db"], np.full(n, 3.0))


def test_overhead_slots_shape():
    sizes = np.array([1, 2, 3, 5, 10])
    result = overhead_slots(sizes)
    assert result["ml_slots"].shape == (5,)
    np.testing.assert_array_equal(result["ml_slots"], N_WIDE_BEAMS + sizes)
    assert 0 < result["mean_reduction"] < 1


def test_overhead_slots_single_beam():
    sizes = np.array([1])
    result = overhead_slots(sizes)
    expected = 1.0 - (N_WIDE_BEAMS + 1) / N_NARROW_BEAMS
    np.testing.assert_almost_equal(result["mean_reduction"], expected)


def test_effective_throughput():
    se = np.array([5.0, 5.0, 5.0])
    overhead = np.array([10, 20, 50])
    result = effective_throughput(se, overhead, total_slots=100)
    assert result["effective_se"][0] > result["effective_se"][1]
    assert result["effective_se"][1] > result["effective_se"][2]


def test_compute_beam_rsrp():
    n_tx = 32
    channels = (np.random.randn(10, n_tx) + 1j * np.random.randn(10, n_tx)) / np.sqrt(n_tx)
    codebook = np.eye(N_NARROW_BEAMS, n_tx, dtype=complex)
    rsrp = compute_beam_rsrp(channels[:, :N_NARROW_BEAMS], codebook)
    assert rsrp.shape == (10, N_NARROW_BEAMS)
