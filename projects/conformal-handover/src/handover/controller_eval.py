from dataclasses import dataclass

import numpy as np

from src.handover.baseline import count_handovers, count_ping_pong


@dataclass
class ReplayConfig:
    a3_offset_db: float = 3.0
    ttt_steps: int = 2
    min_dwell_steps: int = 2
    k_max: int = 12
    interruption_margin_db: float = 6.0


def select_prediction_target(
    rsrp_t: np.ndarray,
    current_serving: int,
    prediction_set: np.ndarray | None,
    top1_prediction: int | None,
    config: ReplayConfig,
):
    n_cells = len(rsrp_t)
    full_scan_target = int(np.argmax(rsrp_t))
    if prediction_set is None:
        if top1_prediction is None:
            return full_scan_target, n_cells, True
        return int(top1_prediction), 1, False
    pred = np.asarray(prediction_set, dtype=int)
    if len(pred) == 0:
        return full_scan_target, n_cells, True
    if len(pred) > config.k_max:
        pred = pred[: config.k_max]
    best_local = int(pred[np.argmax(rsrp_t[pred])])
    return best_local, len(pred), False


def simulate_replay_sequence(
    rsrp_sequence: np.ndarray,
    optimal_sequence: np.ndarray,
    initial_serving: int,
    config: ReplayConfig,
    prediction_sets: list | None = None,
    top1_predictions: np.ndarray | None = None,
    policy: str = "a3",
):
    t_total, n_cells = rsrp_sequence.shape
    serving = np.zeros(t_total, dtype=int)
    serving[0] = int(initial_serving)
    measurement_count = 0.0
    pending_target = None
    pending_steps = 0
    last_handover_step = 0
    successful_handovers = 0

    for t in range(1, t_total):
        current_serving = int(serving[t - 1])
        rsrp_t = rsrp_sequence[t]

        if policy == "a3":
            target = int(np.argmax(rsrp_t))
            measured_cells = n_cells
        else:
            prediction_set = None if prediction_sets is None else prediction_sets[t]
            top1_pred = None if top1_predictions is None else int(top1_predictions[t])
            target, measured_cells, _ = select_prediction_target(
                rsrp_t,
                current_serving,
                prediction_set,
                top1_pred,
                config,
            )

        measurement_count += measured_cells
        serving_rsrp = float(rsrp_t[current_serving])
        target_rsrp = float(rsrp_t[target])
        margin = target_rsrp - serving_rsrp

        valid_switch = (
            target != current_serving
            and margin >= config.a3_offset_db
            and (t - last_handover_step) >= config.min_dwell_steps
        )

        if valid_switch:
            if pending_target == target:
                pending_steps += 1
            else:
                pending_target = target
                pending_steps = 1
        else:
            pending_target = None
            pending_steps = 0

        next_serving = current_serving
        if pending_target is not None and pending_steps >= config.ttt_steps:
            next_serving = int(pending_target)
            last_handover_step = t
            pending_target = None
            pending_steps = 0
            if next_serving == int(optimal_sequence[t]):
                successful_handovers += 1

        serving[t] = next_serving

    best_rsrp = np.max(rsrp_sequence, axis=1)
    serving_rsrp = rsrp_sequence[np.arange(t_total), serving]
    wrong_mask = serving != optimal_sequence
    interruption_mask = (best_rsrp - serving_rsrp) > config.interruption_margin_db
    handover_count = count_handovers(serving)
    ping_pong_count = count_ping_pong(serving)
    decision_steps = max(t_total - 1, 1)

    return {
        "serving_sequence": serving,
        "handover_count": int(handover_count),
        "handover_rate": float(handover_count / decision_steps),
        "handover_precision": float(successful_handovers / max(handover_count, 1)),
        "ping_pong_rate": float(ping_pong_count / max(handover_count, 1)),
        "wrong_cell_dwell": float(np.mean(wrong_mask)),
        "ho_success": float(1.0 - np.mean(wrong_mask)),
        "interruption_proxy": float(np.mean(interruption_mask)),
        "measurement_load": float(measurement_count / (decision_steps * n_cells)),
    }


def evaluate_policy_over_traces(
    rsrp: np.ndarray,
    serving_cell: np.ndarray,
    optimal_cell: np.ndarray,
    trajectory_id: np.ndarray,
    time_step: np.ndarray,
    config: ReplayConfig,
    prediction_sets: list | None = None,
    top1_predictions: np.ndarray | None = None,
    policy: str = "a3",
):
    order = np.lexsort((time_step, trajectory_id))
    rsrp = rsrp[order]
    serving_cell = serving_cell[order]
    optimal_cell = optimal_cell[order]
    trajectory_id = trajectory_id[order]
    if top1_predictions is not None:
        top1_predictions = top1_predictions[order]
    if prediction_sets is not None:
        prediction_sets = [prediction_sets[i] for i in order]

    total = {
        "handover_count": 0.0,
        "ping_pong_rate_weighted": 0.0,
        "handover_precision_weighted": 0.0,
        "wrong_cell_dwell_steps": 0.0,
        "interruption_steps": 0.0,
        "measurement_steps": 0.0,
        "decision_steps": 0.0,
    }

    for traj in np.unique(trajectory_id):
        mask = trajectory_id == traj
        pred_sets = None
        if prediction_sets is not None:
            pred_sets = [prediction_sets[i] for i in np.where(mask)[0]]
        top1 = None if top1_predictions is None else top1_predictions[mask]
        result = simulate_replay_sequence(
            rsrp_sequence=rsrp[mask],
            optimal_sequence=optimal_cell[mask],
            initial_serving=int(serving_cell[mask][0]),
            config=config,
            prediction_sets=pred_sets,
            top1_predictions=top1,
            policy=policy,
        )
        trace_steps = max(int(mask.sum()) - 1, 1)
        total["handover_count"] += result["handover_count"]
        total["ping_pong_rate_weighted"] += result["ping_pong_rate"] * trace_steps
        total["handover_precision_weighted"] += result["handover_precision"] * trace_steps
        total["wrong_cell_dwell_steps"] += result["wrong_cell_dwell"] * mask.sum()
        total["interruption_steps"] += result["interruption_proxy"] * mask.sum()
        total["measurement_steps"] += result["measurement_load"] * trace_steps
        total["decision_steps"] += trace_steps

    sample_steps = max(len(rsrp), 1)
    decision_steps = max(total["decision_steps"], 1.0)
    wrong_cell_dwell = float(total["wrong_cell_dwell_steps"] / sample_steps)
    interruption_proxy = float(total["interruption_steps"] / sample_steps)

    return {
        "handover_count": float(total["handover_count"]),
        "handover_rate": float(total["handover_count"] / decision_steps),
        "handover_precision": float(total["handover_precision_weighted"] / decision_steps),
        "ping_pong_rate": float(total["ping_pong_rate_weighted"] / decision_steps),
        "wrong_cell_dwell": wrong_cell_dwell,
        "ho_success": float(1.0 - wrong_cell_dwell),
        "interruption_proxy": interruption_proxy,
        "measurement_load": float(total["measurement_steps"] / decision_steps),
    }
