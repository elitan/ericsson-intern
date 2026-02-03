"""
3dB threshold baseline for handover prediction.

Traditional rule: handover to strongest cell if it exceeds serving by >3dB.
"""

import numpy as np


def predict_3db_baseline(
    rsrp: np.ndarray,
    serving_cell: np.ndarray,
    threshold_db: float = 3.0,
) -> np.ndarray:
    """
    3dB threshold baseline prediction.

    Args:
        rsrp: (n_samples, n_cells) RSRP measurements
        serving_cell: (n_samples,) current serving cell
        threshold_db: hysteresis threshold (default 3dB)

    Returns:
        (n_samples,) predicted target cell
    """
    n_samples = len(serving_cell)
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        serving_rsrp = rsrp[i, serving_cell[i]]
        best_cell = rsrp[i].argmax()
        best_rsrp = rsrp[i, best_cell]

        if best_rsrp - serving_rsrp > threshold_db:
            predictions[i] = best_cell
        else:
            predictions[i] = serving_cell[i]

    return predictions


def evaluate_3db_baseline(
    rsrp: np.ndarray,
    serving_cell: np.ndarray,
    optimal_cell: np.ndarray,
    threshold_db: float = 3.0,
) -> dict:
    """
    Evaluate 3dB baseline against optimal cell ground truth.

    Returns:
        dict with accuracy, handover_rate, correct_ho_rate
    """
    predictions = predict_3db_baseline(rsrp, serving_cell, threshold_db)

    accuracy = (predictions == optimal_cell).mean()

    ho_triggered = predictions != serving_cell
    ho_rate = ho_triggered.mean()

    correct_ho = (predictions == optimal_cell) & ho_triggered
    correct_ho_rate = correct_ho.sum() / max(ho_triggered.sum(), 1)

    needs_ho = optimal_cell != serving_cell
    ho_when_needed = ho_triggered & needs_ho
    missed_ho = needs_ho & ~ho_triggered
    unnecessary_ho = ho_triggered & ~needs_ho

    return {
        "accuracy": float(accuracy),
        "ho_rate": float(ho_rate),
        "correct_ho_rate": float(correct_ho_rate),
        "missed_ho_rate": float(missed_ho.mean()),
        "unnecessary_ho_rate": float(unnecessary_ho.mean()),
    }


def simulate_3db_handover_sequence(
    rsrp_sequence: np.ndarray,
    initial_serving: int,
    threshold_db: float = 3.0,
) -> np.ndarray:
    """
    Simulate serving cell evolution using 3dB rule.

    Args:
        rsrp_sequence: (T, n_cells) RSRP over time
        initial_serving: starting serving cell
        threshold_db: hysteresis threshold

    Returns:
        (T,) serving cell sequence
    """
    T = len(rsrp_sequence)
    serving = np.zeros(T, dtype=int)
    serving[0] = initial_serving

    for t in range(1, T):
        current_rsrp = rsrp_sequence[t, serving[t - 1]]
        best_cell = rsrp_sequence[t].argmax()
        best_rsrp = rsrp_sequence[t, best_cell]

        if best_rsrp - current_rsrp > threshold_db:
            serving[t] = best_cell
        else:
            serving[t] = serving[t - 1]

    return serving


def count_ping_pong(serving_sequence: np.ndarray, window: int = 3) -> int:
    """
    Count ping-pong handovers (A→B→A patterns).

    Args:
        serving_sequence: (T,) serving cell over time
        window: max steps for A→B→A pattern (default 3)

    Returns:
        number of ping-pong events
    """
    count = 0
    T = len(serving_sequence)

    for i in range(T - 2):
        if (serving_sequence[i] == serving_sequence[i + 2] and
                serving_sequence[i] != serving_sequence[i + 1]):
            count += 1

    return count


def count_handovers(serving_sequence: np.ndarray) -> int:
    """Count total handovers in sequence."""
    return int(np.sum(np.diff(serving_sequence) != 0))


def simulate_policy_serving_sequence(
    rsrp_sequence: np.ndarray,
    initial_serving: int,
    policy: str,
    ml_predictions: np.ndarray = None,
    prediction_sets: list = None,
    threshold_db: float = 3.0,
    k_max: int = 5,
) -> np.ndarray:
    """
    Simulate serving cell evolution under different policies.

    Args:
        rsrp_sequence: (T, n_cells) RSRP over time
        initial_serving: starting serving cell
        policy: "3db", "ml_top1", or "cp_adaptive"
        ml_predictions: (T,) ML top-1 predictions (for ml_top1 policy)
        prediction_sets: list of T prediction sets (for cp_adaptive policy)
        threshold_db: hysteresis threshold for 3dB policy
        k_max: max set size for predictive HO in cp_adaptive

    Returns:
        (T,) serving cell sequence under this policy
    """
    T = len(rsrp_sequence)
    serving = np.zeros(T, dtype=int)
    serving[0] = initial_serving

    for t in range(1, T):
        current_serving = serving[t - 1]
        current_rsrp = rsrp_sequence[t]

        if policy == "3db":
            best_cell = current_rsrp.argmax()
            best_rsrp = current_rsrp[best_cell]
            serving_rsrp = current_rsrp[current_serving]
            if best_rsrp - serving_rsrp > threshold_db:
                serving[t] = best_cell
            else:
                serving[t] = current_serving

        elif policy == "ml_top1":
            serving[t] = ml_predictions[t]

        elif policy == "cp_adaptive":
            pred_set = prediction_sets[t]
            set_size = len(pred_set)

            if set_size <= k_max:
                if current_serving in pred_set:
                    serving[t] = current_serving
                else:
                    set_rsrp = [(c, current_rsrp[c]) for c in pred_set]
                    best_in_set = max(set_rsrp, key=lambda x: x[1])[0]
                    serving[t] = best_in_set
            else:
                best_cell = current_rsrp.argmax()
                best_rsrp = current_rsrp[best_cell]
                serving_rsrp = current_rsrp[current_serving]
                if best_rsrp - serving_rsrp > threshold_db:
                    serving[t] = best_cell
                else:
                    serving[t] = current_serving
        else:
            raise ValueError(f"Unknown policy: {policy}")

    return serving


def compute_ping_pong_metrics(serving_sequence: np.ndarray) -> dict:
    """Compute ping-pong and handover metrics for a serving sequence."""
    n_handovers = count_handovers(serving_sequence)
    n_ping_pong = count_ping_pong(serving_sequence)
    pp_rate = n_ping_pong / max(n_handovers, 1)

    return {
        "n_handovers": n_handovers,
        "n_ping_pong": n_ping_pong,
        "ping_pong_rate": pp_rate,
        "ho_rate": n_handovers / len(serving_sequence),
    }
