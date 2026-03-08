import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_shift_reliability_v7 import (
    build_fixed_k_sets,
    build_recalibration_sets,
    build_trace_split,
    make_regime_switch_data,
    truncate_prediction_sets,
)
from src.handover.conformal import (
    BudgetedDynamicAdaptiveConformalInference,
    DelayedAdaptiveConformalInference,
)
from src.handover.controller_eval import ReplayConfig, select_prediction_target, simulate_replay_sequence


def test_a3_ttt_waits_before_switch():
    rsrp = np.array(
        [
            [-70.0, -74.0],
            [-72.0, -67.0],
            [-73.0, -66.0],
            [-74.0, -65.0],
        ]
    )
    optimal = np.array([0, 1, 1, 1])
    config = ReplayConfig(a3_offset_db=3.0, ttt_steps=2, min_dwell_steps=1, k_max=2)
    result = simulate_replay_sequence(rsrp, optimal, initial_serving=0, config=config, policy="a3")
    assert result["serving_sequence"].tolist() == [0, 0, 1, 1]
    assert result["handover_count"] == 1


def test_delayed_aci_defers_alpha_update():
    aci = DelayedAdaptiveConformalInference(alpha=0.1, gamma=0.2, feedback_delay=2)
    aci.observe(False)
    aci.observe(False)
    assert aci.alpha_t == 0.1
    aci.observe(False)
    assert aci.alpha_t < 0.1


def test_budgeted_daci_pushes_alpha_up_for_large_sets():
    daci = BudgetedDynamicAdaptiveConformalInference(
        alpha=0.1,
        gamma_low=0.01,
        gamma_high=0.02,
        ema_beta=0.9,
        feedback_delay=0,
        target_size_ratio=0.1,
        budget_gamma=0.5,
    )
    daci.observe(True, set_size=8, n_classes=10)
    assert daci.alpha_t > 0.1


def test_recalibration_warm_start_masks_prefix():
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
        ]
    )
    labels = np.array([0, 1, 2, 0])
    order = np.array([0, 1, 2, 3])
    cal_scores = np.array([0.1, 0.2, 0.15, 0.05])
    sets, eval_mask = build_recalibration_sets(
        probs,
        labels,
        order,
        cal_scores,
        alpha=0.1,
        strategy="warm-start",
        budget=2,
        feedback_delay=0,
    )
    assert eval_mask.tolist() == [False, False, True, True]
    assert len(sets[2]) >= 1


def test_prediction_target_respects_k_max():
    config = ReplayConfig(k_max=2)
    target, measured, full_scan = select_prediction_target(
        rsrp_t=np.array([-80.0, -72.0, -65.0, -60.0]),
        current_serving=0,
        prediction_set=np.array([1, 2, 3]),
        top1_prediction=3,
        config=config,
    )
    assert target == 2
    assert measured == 2
    assert full_scan is False


def test_regime_switch_keeps_sequence_structure():
    data = make_regime_switch_data(seed=42, n_traj=20)
    assert len(np.unique(data["trajectory_id"])) > 1
    for traj in np.unique(data["trajectory_id"])[:3]:
        steps = data["time_step"][data["trajectory_id"] == traj]
        assert steps.tolist() == list(range(len(steps)))


def test_chronological_split_keeps_latest_traces_for_target():
    trace_ids = np.repeat(np.arange(6), 2)
    speed_col = np.linspace(0.0, 1.0, len(trace_ids))
    trace_start_map = {i: f"2019.12.0{i + 1}_00.00.00" for i in range(6)}
    split = build_trace_split(
        trace_ids=trace_ids,
        speed_col=speed_col,
        seed=42,
        protocol="chronological-holdout",
        trace_start_map=trace_start_map,
    )
    assert split["target_traces"] == {5}


def test_static_truncation_keeps_top_probs_inside_set():
    probs = np.array([[0.4, 0.3, 0.2, 0.1]])
    sets = [np.array([3, 1, 0, 2])]
    truncated = truncate_prediction_sets(sets, probs, k_max=2)
    assert truncated[0].tolist() == [0, 1]
    fixed_k = build_fixed_k_sets(probs, k_max=2)
    assert fixed_k[0].tolist() == [0, 1]
