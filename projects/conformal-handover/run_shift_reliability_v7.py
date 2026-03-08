import argparse
import datetime
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_irish_experiment import train_model, get_probs
from src.handover.baseline import count_handovers, count_ping_pong
from src.handover.conformal import (
    BudgetedDynamicAdaptiveConformalInference,
    DelayedAdaptiveConformalInference,
    DelayedDynamicAdaptiveConformalInference,
    calibrate_raps,
    calibrate_threshold,
    evaluate_cp,
    predict_sets,
    predict_sets_raps,
)
from src.handover.controller_eval import ReplayConfig, evaluate_policy_over_traces
from src.handover.irish_data import load_driving_traces, preprocess_for_handover
from src.handover.predictor import train_predictor
from src.handover.synthetic_data import MobilityConfig, NetworkConfig, generate_dataset


@dataclass
class ShiftConfig:
    name: str
    noise_std_db: float
    measurement_noise_db: float
    speed_min: float
    speed_max: float
    cell_radius: float = 150.0
    prediction_horizon: int = 10
    stale_neighbor_fraction: float = 0.0
    stale_noise_db: float = 0.0


SOURCE_SHIFT = ShiftConfig("Source", 6.0, 4.0, 1.0, 30.0)

SHIFT_CONFIGS = {
    "iid": SOURCE_SHIFT,
    "speed-shift": ShiftConfig("Speed Shift", 6.0, 4.0, 20.0, 50.0),
    "shadow-shift": ShiftConfig("Shadow Shift", 10.0, 4.0, 1.0, 30.0),
    "horizon-shift": ShiftConfig("Horizon Shift", 6.0, 4.0, 1.0, 30.0, prediction_horizon=20),
    "stale-neighbor-shift": ShiftConfig(
        "Stale Neighbor Shift",
        6.0,
        4.0,
        1.0,
        30.0,
        stale_neighbor_fraction=0.5,
        stale_noise_db=8.0,
    ),
}

SHIFT_ORDER = [
    "iid",
    "speed-shift",
    "shadow-shift",
    "horizon-shift",
    "stale-neighbor-shift",
    "regime-switch",
]

METHOD_ORDER = [
    "a3-ttt",
    "static-cp",
    "raps-cp",
    "triggered-aci",
    "daci",
    "budgeted-daci",
    "fixed-k",
    "static-cp-trunc",
]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit(project_dir: Path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_dir, text=True).strip()
    except Exception:
        return "unknown"


def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def budget_label(budget: int):
    return "full" if int(budget) == 0 else str(int(budget))


def budget_sweep_key(target_size_ratio: float, budget_gamma: float):
    return f"target={target_size_ratio:.3f}|gamma={budget_gamma:.3f}"


def split_by_trajectory(traj_ids: np.ndarray, n_traj: int):
    train_traj = set(range(int(0.6 * n_traj)))
    cal_traj = set(range(int(0.6 * n_traj), int(0.8 * n_traj)))
    test_traj = set(range(int(0.8 * n_traj), n_traj))
    train_idx = np.array([i for i, t in enumerate(traj_ids) if t in train_traj])
    cal_idx = np.array([i for i, t in enumerate(traj_ids) if t in cal_traj])
    test_idx = np.array([i for i, t in enumerate(traj_ids) if t in test_traj])
    return train_idx, cal_idx, test_idx


def source_stats(data: dict):
    rsrp = data["rsrp"].astype(np.float32)
    speed = data["ue_speed"].astype(np.float32)
    return {
        "rsrp_mean": float(rsrp.mean()),
        "rsrp_std": float(rsrp.std() + 1e-8),
        "speed_mean": float(speed.mean()),
        "speed_std": float(speed.std() + 1e-8),
    }


def normalize_features(data: dict, idx: np.ndarray, stats: dict):
    rsrp = data["rsrp"][idx].astype(np.float32)
    speed = data["ue_speed"][idx].astype(np.float32)
    rsrp = (rsrp - stats["rsrp_mean"]) / stats["rsrp_std"]
    speed = (speed - stats["speed_mean"]) / stats["speed_std"]
    return rsrp, speed


def probs_with_stats(model, data: dict, idx: np.ndarray, stats: dict, device: str):
    rsrp, speed = normalize_features(data, idx, stats)
    serving = data["serving_cell"][idx]
    with torch.no_grad():
        rsrp_t = torch.tensor(rsrp, dtype=torch.float32, device=device)
        serving_t = torch.tensor(serving, dtype=torch.long, device=device)
        speed_t = torch.tensor(speed, dtype=torch.float32, device=device)
        logits = model(rsrp_t, serving_t, speed_t)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def aggregate_seed_metrics(seed_metrics: list):
    keys = seed_metrics[0].keys()
    out = {}
    for key in keys:
        vals = np.array([metric[key] for metric in seed_metrics], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            half = 1.96 * sem
        else:
            half = 0.0
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
        out[f"{key}_ci95_low"] = mean - half
        out[f"{key}_ci95_high"] = mean + half
    return out


def paired_delta_with_ci(seed_a: list, seed_b: list, field: str, n_boot: int = 2000):
    a = np.array([x[field] for x in seed_a], dtype=float)
    b = np.array([x[field] for x in seed_b], dtype=float)
    delta = a - b
    rng = np.random.default_rng(123)
    boot = []
    n = len(delta)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(delta[idx].mean())
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5])
    return {
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def make_decile_bins(values: np.ndarray):
    n = len(values)
    order = np.argsort(values)
    bins = np.zeros(n, dtype=int)
    bins[order] = (np.arange(n) * 10) // max(n, 1)
    bins = np.clip(bins, 0, 9)
    return bins


def conditional_summary(values: np.ndarray, metric_arrays: dict):
    bins = make_decile_bins(values)
    out = {"deciles": []}
    for decile in range(10):
        idx = np.where(bins == decile)[0]
        row = {"decile": decile, "n": int(len(idx)), "methods": {}}
        for method, arr in metric_arrays.items():
            if len(idx) == 0:
                row["methods"][method] = float("nan")
            else:
                row["methods"][method] = float(np.mean(arr[idx]))
        out["deciles"].append(row)
    return out


def aggregate_conditional_summaries(seed_summaries: list):
    domains = list(seed_summaries[0].keys())
    out = {}
    for domain in domains:
        methods = list(seed_summaries[0][domain]["deciles"][0]["methods"].keys())
        rows = []
        for decile in range(10):
            row = {"decile": decile, "methods": {}}
            row["n_mean"] = float(np.mean([s[domain]["deciles"][decile]["n"] for s in seed_summaries]))
            for method in methods:
                vals = [s[domain]["deciles"][decile]["methods"][method] for s in seed_summaries]
                row["methods"][method] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
            rows.append(row)
        out[domain] = rows
    return out


def ensure_nonempty_set(indices: np.ndarray, probs_row: np.ndarray):
    if len(indices) == 0:
        return np.array([int(np.argmax(probs_row))])
    return indices


def set_size_array(prediction_sets: list):
    return np.array([len(item) for item in prediction_sets], dtype=float)


def set_cover_array(labels: np.ndarray, prediction_sets: list):
    return np.array([int(labels[i] in prediction_sets[i]) for i in range(len(labels))], dtype=float)


def build_fixed_k_sets(probs: np.ndarray, k_max: int):
    sets = []
    keep = max(int(k_max), 1)
    for row in probs:
        order = np.argsort(row)[::-1][:keep]
        sets.append(np.array(order, dtype=int))
    return sets


def truncate_prediction_sets(prediction_sets: list, probs: np.ndarray, k_max: int):
    keep = max(int(k_max), 1)
    out = []
    for i, prediction_set in enumerate(prediction_sets):
        pred = np.asarray(prediction_set, dtype=int)
        if len(pred) <= keep:
            out.append(pred)
            continue
        order = np.argsort(probs[i, pred])[::-1][:keep]
        out.append(pred[order])
    return out


def trace_bootstrap_ci(trace_ids: np.ndarray, cover: np.ndarray, sizes: np.ndarray, n_boot: int = 2000, seed: int = 123):
    uniq = np.unique(trace_ids)
    if len(uniq) == 0:
        raise ValueError("empty trace ids")
    idx_by_trace = {int(trace): np.where(trace_ids == trace)[0] for trace in uniq}
    rng = np.random.default_rng(seed)
    cover_boot = []
    size_boot = []
    uniq_list = [int(trace) for trace in uniq.tolist()]
    for _ in range(n_boot):
        sampled = rng.choice(uniq_list, size=len(uniq_list), replace=True)
        idx = np.concatenate([idx_by_trace[int(trace)] for trace in sampled])
        cover_boot.append(float(np.mean(cover[idx])))
        size_boot.append(float(np.mean(sizes[idx])))
    c_lo, c_hi = np.percentile(np.array(cover_boot), [2.5, 97.5])
    s_lo, s_hi = np.percentile(np.array(size_boot), [2.5, 97.5])
    return {
        "coverage_ci95_low": float(c_lo),
        "coverage_ci95_high": float(c_hi),
        "avg_set_size_ci95_low": float(s_lo),
        "avg_set_size_ci95_high": float(s_hi),
    }


def trace_bootstrap_delta_ci(
    trace_ids: np.ndarray,
    cover_a: np.ndarray,
    cover_b: np.ndarray,
    sizes_a: np.ndarray,
    sizes_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 123,
):
    uniq = np.unique(trace_ids)
    if len(uniq) == 0:
        raise ValueError("empty trace ids")
    idx_by_trace = {int(trace): np.where(trace_ids == trace)[0] for trace in uniq}
    rng = np.random.default_rng(seed)
    cover_boot = []
    size_boot = []
    uniq_list = [int(trace) for trace in uniq.tolist()]
    for _ in range(n_boot):
        sampled = rng.choice(uniq_list, size=len(uniq_list), replace=True)
        idx = np.concatenate([idx_by_trace[int(trace)] for trace in sampled])
        cover_boot.append(float(np.mean(cover_a[idx]) - np.mean(cover_b[idx])))
        size_boot.append(float(np.mean(sizes_a[idx]) - np.mean(sizes_b[idx])))
    c_lo, c_hi = np.percentile(np.array(cover_boot), [2.5, 97.5])
    s_lo, s_hi = np.percentile(np.array(size_boot), [2.5, 97.5])
    return {
        "coverage_delta_mean": float(np.mean(cover_a) - np.mean(cover_b)),
        "coverage_ci95_low": float(c_lo),
        "coverage_ci95_high": float(c_hi),
        "avg_set_size_delta_mean": float(np.mean(sizes_a) - np.mean(sizes_b)),
        "avg_set_size_ci95_low": float(s_lo),
        "avg_set_size_ci95_high": float(s_hi),
    }


def aggregate_named_fields(items: list, fields: list[str]):
    out = {}
    for field in fields:
        vals = np.array([item[field] for item in items], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            half = 1.96 * sem
        else:
            half = 0.0
        out[f"{field}_mean"] = mean
        out[f"{field}_std"] = std
        out[f"{field}_ci95_low"] = mean - half
        out[f"{field}_ci95_high"] = mean + half
    return out


def summarize_worst_deciles(runs: list[dict]):
    domains = list(runs[0].keys())
    out = {}
    for domain in domains:
        methods = list(runs[0][domain]["deciles"][0]["methods"].keys())
        out[domain] = {}
        for method in methods:
            min_vals = []
            mean_vals = []
            worst_idx_vals = []
            for run in runs:
                vals = np.array(
                    [row["methods"][method] for row in run[domain]["deciles"]],
                    dtype=float,
                )
                worst_idx = int(np.argmin(vals))
                worst_idx_vals.append(worst_idx)
                min_vals.append(float(vals[worst_idx]))
                mean_vals.append(float(np.mean(vals)))
            min_stats = aggregate_named_fields(
                [{"value": value} for value in min_vals],
                ["value"],
            )
            mean_stats = aggregate_named_fields(
                [{"value": value} for value in mean_vals],
                ["value"],
            )
            mode_idx = int(np.bincount(np.array(worst_idx_vals, dtype=int)).argmax())
            out[domain][method] = {
                "worst_decile_mode": mode_idx,
                "min_coverage": float(min_stats["value_mean"]),
                "min_coverage_std": float(min_stats["value_std"]),
                "min_coverage_ci95_low": float(min_stats["value_ci95_low"]),
                "min_coverage_ci95_high": float(min_stats["value_ci95_high"]),
                "mean_coverage": float(mean_stats["value_mean"]),
                "mean_coverage_std": float(mean_stats["value_std"]),
                "mean_coverage_ci95_low": float(mean_stats["value_ci95_low"]),
                "mean_coverage_ci95_high": float(mean_stats["value_ci95_high"]),
            }
    return out


def pick_best_budget_config(config_results: dict, max_load: float | None = None):
    candidates = []
    for key, metrics in config_results.items():
        load = float(metrics["measurement_load_mean"])
        if max_load is not None and load > max_load:
            continue
        candidates.append((key, metrics))
    if not candidates:
        return None
    best_key, best_metrics = max(
        candidates,
        key=lambda item: (
            float(item[1]["coverage_mean"]),
            -float(item[1]["measurement_load_mean"]),
        ),
    )
    out = {"key": best_key}
    out.update(best_metrics)
    return out


def pareto_frontier_methods(results: dict, method_names: list[str], y_key: str = "coverage", x_key: str = "measurement_load"):
    frontier = []
    for method in method_names:
        dominated = False
        y_val = float(results[method][f"{y_key}_mean"] if f"{y_key}_mean" in results[method] else results[method][y_key])
        x_val = float(results[method][f"{x_key}_mean"] if f"{x_key}_mean" in results[method] else results[method][x_key])
        for other in method_names:
            if other == method:
                continue
            other_y = float(results[other][f"{y_key}_mean"] if f"{y_key}_mean" in results[other] else results[other][y_key])
            other_x = float(results[other][f"{x_key}_mean"] if f"{x_key}_mean" in results[other] else results[other][x_key])
            if other_x <= x_val and other_y >= y_val and (other_x < x_val or other_y > y_val):
                dominated = True
                break
        if not dominated:
            frontier.append(method)
    return frontier


def aggregate_budget_sweep_runs(runs: list[dict], daci_load: float):
    keys = list(runs[0].keys())
    config_results = {}
    for key in keys:
        vals = runs[0][key]
        aggregated = aggregate_named_fields(
            [run[key] for run in runs],
            ["coverage", "avg_set_size", "measurement_load", "worst_confidence_min"],
        )
        aggregated["budget_target"] = float(vals["budget_target"])
        aggregated["budget_gamma"] = float(vals["budget_gamma"])
        config_results[key] = aggregated
    frontier_configs = pareto_frontier_methods(
        config_results,
        list(config_results.keys()),
        y_key="coverage",
        x_key="measurement_load",
    )
    return {
        "configs": config_results,
        "frontier_configs": frontier_configs,
        "best_overall": pick_best_budget_config(config_results),
        "best_under_daci_load": pick_best_budget_config(config_results, max_load=daci_load),
    }


def ordered_quantile(scores: np.ndarray, alpha_t: float):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha_t)) / n
    q_level = np.clip(q_level, 0, 1)
    return float(np.quantile(scores, q_level, method="higher"))


def prediction_set_from_threshold(probs_row: np.ndarray, threshold: float):
    included = np.where(probs_row >= 1 - threshold)[0]
    included = ensure_nonempty_set(included, probs_row)
    order = np.argsort(probs_row[included])[::-1]
    return included[order]


def apply_neighbor_shift(data: dict, seed: int, fraction: float, noise_db: float):
    if fraction <= 0.0:
        return data
    rng = np.random.default_rng(seed)
    shifted = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in data.items()}
    rsrp = shifted["rsrp"].copy()
    serving = shifted["serving_cell"]
    mask = rng.random(rsrp.shape) < fraction
    row_idx = np.arange(rsrp.shape[0])[:, None]
    serving_mask = np.zeros_like(mask)
    serving_mask[row_idx, serving[:, None]] = True
    mask = mask & (~serving_mask)
    stale_values = rsrp[row_idx, serving[:, None]] - rng.normal(noise_db, 1.5, size=rsrp.shape)
    rsrp[mask] = stale_values[mask]
    shifted["rsrp"] = rsrp
    return shifted


def generate_shift_data(seed: int, n_traj: int, cfg: ShiftConfig):
    network = NetworkConfig(
        n_gnb_x=4,
        n_gnb_y=4,
        cell_radius=cfg.cell_radius,
        noise_std_db=cfg.noise_std_db,
    )
    mobility = MobilityConfig(
        n_trajectories=n_traj,
        trajectory_length=100,
        speed_range=(cfg.speed_min, cfg.speed_max),
    )
    data = generate_dataset(
        network,
        mobility,
        seed=seed,
        prediction_horizon=cfg.prediction_horizon,
        measurement_noise_db=cfg.measurement_noise_db,
    )
    return apply_neighbor_shift(data, seed + 9000, cfg.stale_neighbor_fraction, cfg.stale_noise_db)


def make_regime_switch_data(seed: int, n_traj: int):
    data_a = generate_shift_data(seed, n_traj, SHIFT_CONFIGS["iid"])
    harsh = ShiftConfig(
        "Regime Harsh",
        10.0,
        4.0,
        20.0,
        50.0,
        prediction_horizon=20,
        stale_neighbor_fraction=0.35,
        stale_noise_db=6.0,
    )
    data_b = generate_shift_data(seed + 2000, n_traj, harsh)
    _, _, test_a = split_by_trajectory(data_a["trajectory_id"], n_traj)
    _, _, test_b = split_by_trajectory(data_b["trajectory_id"], n_traj)
    test_traj_a = np.unique(data_a["trajectory_id"][test_a])
    test_traj_b = np.unique(data_b["trajectory_id"][test_b])
    n_pair = min(len(test_traj_a), len(test_traj_b))
    rsrp = []
    serving = []
    optimal = []
    speed = []
    trajectory_id = []
    time_step = []
    phase_boundary = 0

    for new_traj, (traj_a, traj_b) in enumerate(zip(test_traj_a[:n_pair], test_traj_b[:n_pair])):
        mask_a = data_a["trajectory_id"] == traj_a
        mask_b = data_b["trajectory_id"] == traj_b
        order_a = np.argsort(data_a["time_step"][mask_a])
        order_b = np.argsort(data_b["time_step"][mask_b])
        rsrp_a = data_a["rsrp"][mask_a][order_a]
        rsrp_b = data_b["rsrp"][mask_b][order_b]
        serving_a = data_a["serving_cell"][mask_a][order_a]
        serving_b = data_b["serving_cell"][mask_b][order_b]
        optimal_a = data_a["optimal_cell"][mask_a][order_a]
        optimal_b = data_b["optimal_cell"][mask_b][order_b]
        speed_a = data_a["ue_speed"][mask_a][order_a]
        speed_b = data_b["ue_speed"][mask_b][order_b]
        cut = min(len(rsrp_a), len(rsrp_b)) // 2
        phase_boundary = cut
        rsrp.append(np.concatenate([rsrp_a[:cut], rsrp_b[cut:]], axis=0))
        serving.append(np.concatenate([serving_a[:cut], serving_b[cut:]], axis=0))
        optimal.append(np.concatenate([optimal_a[:cut], optimal_b[cut:]], axis=0))
        speed.append(np.concatenate([speed_a[:cut], speed_b[cut:]], axis=0))
        trajectory_id.append(np.full(len(rsrp[-1]), new_traj))
        time_step.append(np.arange(len(rsrp[-1])))

    return {
        "rsrp": np.concatenate(rsrp, axis=0),
        "serving_cell": np.concatenate(serving, axis=0),
        "optimal_cell": np.concatenate(optimal, axis=0),
        "ue_speed": np.concatenate(speed, axis=0),
        "trajectory_id": np.concatenate(trajectory_id),
        "time_step": np.concatenate(time_step),
        "n_cells": int(data_a["n_cells"]),
        "phase_boundary": int(phase_boundary),
    }


def build_static_sets(probs: np.ndarray, threshold: float):
    return [prediction_set_from_threshold(row, threshold) for row in probs]


def build_daci_sets(
    probs: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float,
    gamma_low: float,
    gamma_high: float,
    ema_beta: float,
    feedback_delay: int = 0,
):
    controller = DelayedDynamicAdaptiveConformalInference(
        alpha=alpha,
        gamma_low=gamma_low,
        gamma_high=gamma_high,
        ema_beta=ema_beta,
        feedback_delay=feedback_delay,
    )
    prediction_sets = [None] * len(labels)
    alpha_series = []
    for rel in order:
        threshold = controller.get_threshold(cal_scores)
        prediction_set = prediction_set_from_threshold(probs[rel], threshold)
        prediction_sets[rel] = prediction_set
        controller.observe(labels[rel] in prediction_set)
        alpha_series.append(float(controller.alpha_t))
    return prediction_sets, alpha_series


def build_budgeted_daci_sets(
    probs: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float,
    gamma_low: float,
    gamma_high: float,
    ema_beta: float,
    feedback_delay: int,
    target_size_ratio: float,
    budget_gamma: float,
    n_classes: int,
):
    controller = BudgetedDynamicAdaptiveConformalInference(
        alpha=alpha,
        gamma_low=gamma_low,
        gamma_high=gamma_high,
        ema_beta=ema_beta,
        feedback_delay=feedback_delay,
        target_size_ratio=target_size_ratio,
        budget_gamma=budget_gamma,
    )
    prediction_sets = [None] * len(labels)
    alpha_series = []
    for rel in order:
        threshold = controller.get_threshold(cal_scores)
        prediction_set = prediction_set_from_threshold(probs[rel], threshold)
        prediction_sets[rel] = prediction_set
        controller.observe(labels[rel] in prediction_set, len(prediction_set), n_classes)
        alpha_series.append(float(controller.alpha_t))
    return prediction_sets, alpha_series


def build_triggered_aci_sets(
    probs: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    cal_scores: np.ndarray,
    static_threshold: float,
    source_confidence: np.ndarray,
    alpha: float,
    gamma: float,
    trigger_quantile: float,
    feedback_delay: int = 0,
):
    controller = DelayedAdaptiveConformalInference(alpha=alpha, gamma=gamma, feedback_delay=feedback_delay)
    tau = float(np.quantile(source_confidence, trigger_quantile))
    prediction_sets = [None] * len(labels)
    coverage = np.zeros(len(labels), dtype=float)
    for rel in order:
        dynamic_threshold = controller.get_threshold(cal_scores)
        dynamic_set = prediction_set_from_threshold(probs[rel], dynamic_threshold)
        static_set = prediction_set_from_threshold(probs[rel], static_threshold)
        if float(np.max(probs[rel])) < tau:
            prediction_set = dynamic_set
        else:
            prediction_set = static_set
        prediction_sets[rel] = prediction_set
        covered = bool(labels[rel] in prediction_set)
        coverage[rel] = float(covered)
        controller.observe(covered)
    return prediction_sets, tau, coverage


def build_recalibration_sets(
    probs: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    source_cal_scores: np.ndarray,
    alpha: float,
    strategy: str,
    budget: int,
    feedback_delay: int,
):
    source_scores = list(np.asarray(source_cal_scores, dtype=float))
    prediction_sets = [None] * len(labels)
    evaluation_mask = np.ones(len(labels), dtype=bool)
    pending = []
    recent_target_scores = []

    if strategy == "source-only":
        threshold = ordered_quantile(np.array(source_scores), alpha)
        return build_static_sets(probs, threshold), evaluation_mask

    if strategy == "warm-start":
        for rel in order[:budget]:
            source_scores.append(1 - float(probs[rel, labels[rel]]))
            evaluation_mask[rel] = False
        threshold = ordered_quantile(np.array(source_scores), alpha)
        return build_static_sets(probs, threshold), evaluation_mask

    for rel in order:
        current_scores = source_scores + recent_target_scores
        threshold = ordered_quantile(np.array(current_scores), alpha)
        prediction_sets[rel] = prediction_set_from_threshold(probs[rel], threshold)
        pending.append(1 - float(probs[rel, labels[rel]]))
        if len(pending) > feedback_delay:
            recent_target_scores.append(float(pending.pop(0)))
            if budget > 0 and len(recent_target_scores) > budget:
                recent_target_scores = recent_target_scores[-budget:]

    return prediction_sets, evaluation_mask


def evaluate_method_bundle(
    target_data: dict,
    target_idx: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    source_cal_probs: np.ndarray,
    source_cal_labels: np.ndarray,
    cal_scores: np.ndarray,
    replay_config: ReplayConfig,
    alpha: float,
    gamma: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
    trigger_quantile: float,
    raps_k_reg: int,
    raps_lam: float,
    feedback_delay: int,
    budget_target: float,
    budget_gamma: float,
):
    static_threshold = calibrate_threshold(source_cal_probs, source_cal_labels, alpha=alpha)
    raps_threshold = calibrate_raps(
        source_cal_probs,
        source_cal_labels,
        alpha=alpha,
        k_reg=raps_k_reg,
        lam=raps_lam,
        rand=False,
    )

    order = np.lexsort((target_data["time_step"][target_idx], target_data["trajectory_id"][target_idx]))
    top1_pred = np.argmax(probs, axis=1)
    source_confidence = np.max(source_cal_probs, axis=1)

    method_sets = {
        "static-cp": build_static_sets(probs, static_threshold),
        "raps-cp": predict_sets_raps(
            probs,
            raps_threshold,
            k_reg=raps_k_reg,
            lam=raps_lam,
            rand=False,
        ),
    }
    method_sets["triggered-aci"], trigger_tau, triggered_cover = build_triggered_aci_sets(
        probs,
        labels,
        order,
        cal_scores,
        static_threshold,
        source_confidence,
        alpha,
        gamma,
        trigger_quantile,
        feedback_delay=feedback_delay,
    )
    method_sets["daci"], daci_alpha_series = build_daci_sets(
        probs,
        labels,
        order,
        cal_scores,
        alpha,
        daci_gamma_low,
        daci_gamma_high,
        daci_ema_beta,
        feedback_delay=feedback_delay,
    )
    method_sets["budgeted-daci"], budget_alpha_series = build_budgeted_daci_sets(
        probs,
        labels,
        order,
        cal_scores,
        alpha,
        daci_gamma_low,
        daci_gamma_high,
        daci_ema_beta,
        feedback_delay,
        budget_target,
        budget_gamma,
        probs.shape[1],
    )
    method_sets["fixed-k"] = build_fixed_k_sets(probs, replay_config.k_max)
    method_sets["static-cp-trunc"] = truncate_prediction_sets(method_sets["static-cp"], probs, replay_config.k_max)

    results = {}
    coverage_arrays = {}
    size_arrays = {}
    confidence = np.max(probs, axis=1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    prob_margin = top2[:, 1] - top2[:, 0]
    signal_margin = np.max(target_data["rsrp"][target_idx], axis=1) - target_data["rsrp"][target_idx, target_data["serving_cell"][target_idx]]

    baseline_metrics = evaluate_policy_over_traces(
        target_data["rsrp"][target_idx],
        target_data["serving_cell"][target_idx],
        labels,
        target_data["trajectory_id"][target_idx],
        target_data["time_step"][target_idx],
        replay_config,
        policy="a3",
    )
    results["a3-ttt"] = baseline_metrics
    results["a3-ttt"]["coverage"] = baseline_metrics["ho_success"]
    results["a3-ttt"]["avg_set_size"] = float(probs.shape[1])

    for method, prediction_sets in method_sets.items():
        cp_metrics = evaluate_cp(prediction_sets, labels)
        replay_metrics = evaluate_policy_over_traces(
            target_data["rsrp"][target_idx],
            target_data["serving_cell"][target_idx],
            labels,
            target_data["trajectory_id"][target_idx],
            target_data["time_step"][target_idx],
            replay_config,
            prediction_sets=prediction_sets,
            top1_predictions=top1_pred,
            policy="cp",
        )
        results[method] = {
            "coverage": float(cp_metrics["coverage"]),
            "avg_set_size": float(cp_metrics["avg_set_size"]),
            "set_size_std": float(cp_metrics["set_size_std"]),
            "ho_success": float(replay_metrics["ho_success"]),
            "measurement_load": float(replay_metrics["measurement_load"]),
            "ping_pong_rate": float(replay_metrics["ping_pong_rate"]),
            "handover_rate": float(replay_metrics["handover_rate"]),
            "handover_precision": float(replay_metrics["handover_precision"]),
            "wrong_cell_dwell": float(replay_metrics["wrong_cell_dwell"]),
            "interruption_proxy": float(replay_metrics["interruption_proxy"]),
        }
        coverage_arrays[method] = set_cover_array(labels, prediction_sets)
        size_arrays[method] = set_size_array(prediction_sets)

    worst_slice = {
        "speed": conditional_summary(target_data["ue_speed"][target_idx], coverage_arrays),
        "confidence": conditional_summary(confidence, coverage_arrays),
        "signal_margin": conditional_summary(signal_margin, coverage_arrays),
        "prob_margin": conditional_summary(prob_margin, coverage_arrays),
    }

    aux = {
        "trigger_tau": trigger_tau,
        "daci_alpha_series_tail": daci_alpha_series[-20:],
        "budgeted_daci_alpha_series_tail": budget_alpha_series[-20:],
        "triggered_cover_mean": float(np.mean(triggered_cover)),
        "coverage_arrays": coverage_arrays,
        "size_arrays": size_arrays,
        "worst_slice": worst_slice,
        "top1_predictions": top1_pred,
        "order": order,
    }
    return results, aux


def evaluate_feedback_delays(
    target_data: dict,
    target_idx: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    source_cal_probs: np.ndarray,
    source_cal_labels: np.ndarray,
    cal_scores: np.ndarray,
    replay_config: ReplayConfig,
    alpha: float,
    gamma: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
    trigger_quantile: float,
    delay_grid: list,
):
    out = {"triggered-aci": {}, "daci": {}}
    for delay in delay_grid:
        method_out, _ = evaluate_method_bundle(
            target_data=target_data,
            target_idx=target_idx,
            labels=labels,
            probs=probs,
            source_cal_probs=source_cal_probs,
            source_cal_labels=source_cal_labels,
            cal_scores=cal_scores,
            replay_config=replay_config,
            alpha=alpha,
            gamma=gamma,
            daci_gamma_low=daci_gamma_low,
            daci_gamma_high=daci_gamma_high,
            daci_ema_beta=daci_ema_beta,
            trigger_quantile=trigger_quantile,
            raps_k_reg=1,
            raps_lam=0.01,
            feedback_delay=delay,
            budget_target=0.12,
            budget_gamma=0.02,
        )
        for method in out.keys():
            out[method][str(delay)] = method_out[method]
    return out


def evaluate_recalibration_budgets(
    target_data: dict,
    target_idx: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    cal_scores: np.ndarray,
    replay_config: ReplayConfig,
    alpha: float,
    budget_grid: list,
    feedback_delay: int,
):
    order = np.lexsort((target_data["time_step"][target_idx], target_data["trajectory_id"][target_idx]))
    out = {"source-only": {}, "rolling-window": {}, "warm-start": {}}
    for budget in budget_grid:
        key = budget_label(budget)
        for strategy in out.keys():
            sets, eval_mask = build_recalibration_sets(
                probs,
                labels,
                order,
                cal_scores,
                alpha,
                strategy=strategy,
                budget=budget,
                feedback_delay=feedback_delay,
            )
            subset_idx = np.where(eval_mask)[0]
            subset_sets = [sets[i] for i in subset_idx]
            subset_labels = labels[subset_idx]
            cp_metrics = evaluate_cp(subset_sets, subset_labels)
            replay_metrics = evaluate_policy_over_traces(
                target_data["rsrp"][target_idx][subset_idx],
                target_data["serving_cell"][target_idx][subset_idx],
                subset_labels,
                target_data["trajectory_id"][target_idx][subset_idx],
                target_data["time_step"][target_idx][subset_idx],
                replay_config,
                prediction_sets=subset_sets,
                top1_predictions=np.argmax(probs[subset_idx], axis=1),
                policy="cp",
            )
            out[strategy][key] = {
                "coverage": float(cp_metrics["coverage"]),
                "avg_set_size": float(cp_metrics["avg_set_size"]),
                "ho_success": float(replay_metrics["ho_success"]),
                "measurement_load": float(replay_metrics["measurement_load"]),
                "wrong_cell_dwell": float(replay_metrics["wrong_cell_dwell"]),
                "interruption_proxy": float(replay_metrics["interruption_proxy"]),
            }
    return out


def evaluate_synthetic_seed(
    seed: int,
    n_traj: int,
    epochs: int,
    batch_size: int,
    alpha: float,
    gamma: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
    trigger_quantile: float,
    replay_config: ReplayConfig,
    device: str,
    delay_grid: list,
    budget_grid: list,
    budget_target: float,
    budget_gamma: float,
):
    set_global_seed(seed)
    source_data = generate_shift_data(seed, n_traj, SOURCE_SHIFT)
    train_idx, cal_idx, _ = split_by_trajectory(source_data["trajectory_id"], n_traj)
    model = train_predictor(
        source_data,
        train_idx,
        cal_idx,
        n_epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    stats = source_stats(source_data)
    source_cal_probs = probs_with_stats(model, source_data, cal_idx, stats, device)
    source_cal_labels = source_data["optimal_cell"][cal_idx]
    cal_scores = 1 - source_cal_probs[np.arange(len(source_cal_labels)), source_cal_labels]

    shift_results = {}
    delayed = {}
    recalibration = {}
    worst_slice_seed = {}

    for shift in SHIFT_ORDER:
        if shift == "regime-switch":
            target_data = make_regime_switch_data(seed, n_traj)
            target_idx = np.arange(len(target_data["optimal_cell"]))
        else:
            target_data = generate_shift_data(seed + 1000 + SHIFT_ORDER.index(shift), n_traj, SHIFT_CONFIGS[shift])
            _, _, target_idx = split_by_trajectory(target_data["trajectory_id"], n_traj)
        probs = probs_with_stats(model, target_data, target_idx, stats, device)
        labels = target_data["optimal_cell"][target_idx]
        method_results, aux = evaluate_method_bundle(
            target_data,
            target_idx,
            labels,
            probs,
            source_cal_probs,
            source_cal_labels,
            cal_scores,
            replay_config,
            alpha,
            gamma,
            daci_gamma_low,
            daci_gamma_high,
            daci_ema_beta,
            trigger_quantile,
            1,
            0.01,
            0,
            budget_target,
            budget_gamma,
        )
        shift_results[shift] = method_results
        worst_slice_seed[shift] = aux["worst_slice"]
        if shift == "regime-switch":
            delayed["regime-switch"] = evaluate_feedback_delays(
                target_data,
                target_idx,
                labels,
                probs,
                source_cal_probs,
                source_cal_labels,
                cal_scores,
                replay_config,
                alpha,
                gamma,
                daci_gamma_low,
                daci_gamma_high,
                daci_ema_beta,
                trigger_quantile,
                delay_grid,
            )
            recalibration["regime-switch"] = evaluate_recalibration_budgets(
                target_data,
                target_idx,
                labels,
                probs,
                cal_scores,
                replay_config,
                alpha,
                budget_grid,
                0,
            )

    return {
        "shift_results": shift_results,
        "delayed_feedback": delayed,
        "recalibration": recalibration,
        "worst_slice": worst_slice_seed,
    }


def ensure_irish_data(project_dir: Path):
    primary = project_dir / "data" / "irish_5g" / "5G-production-dataset"
    if primary.exists():
        return primary
    root = project_dir / "data" / "irish_5g"
    root.mkdir(parents=True, exist_ok=True)
    repo_target = root / "5Gdataset"
    if not repo_target.exists():
        subprocess.run(["gh", "repo", "clone", "uccmisl/5Gdataset", str(repo_target)], check=True)
    zip_path = repo_target / "5G-production-dataset.zip"
    if zip_path.exists() and not primary.exists():
        subprocess.run(["unzip", "-q", "-n", str(zip_path), "-d", str(root)], check=True)
    if primary.exists():
        return primary
    if repo_target.exists():
        return repo_target
    raise RuntimeError("Irish dataset not found")


def build_trace_split(trace_ids: np.ndarray, speed_col: np.ndarray, seed: int, protocol: str, trace_start_map: dict | None = None):
    unique_traces = np.unique(trace_ids)
    if len(unique_traces) < 5:
        raise ValueError("need at least 5 traces for Irish split evaluation")
    if protocol == "speed-split":
        trace_speed = []
        for trace in unique_traces:
            mask = trace_ids == trace
            trace_speed.append((trace, float(speed_col[mask].mean())))
        trace_speed.sort(key=lambda item: item[1])
        half = len(trace_speed) // 2
        source_traces = np.array([trace for trace, _ in trace_speed[:half]])
        target_traces = set([trace for trace, _ in trace_speed[half:]])
        rng = np.random.default_rng(seed)
        rng.shuffle(source_traces)
        split = min(max(int(0.7 * len(source_traces)), 1), max(len(source_traces) - 1, 1))
        train_traces = set(source_traces[:split].tolist())
        cal_traces = set(source_traces[split:].tolist())
    elif protocol == "trace-holdout":
        traces = np.array(unique_traces)
        rng = np.random.default_rng(seed)
        rng.shuffle(traces)
        n = len(traces)
        n_train = min(max(int(0.6 * n), 1), max(n - 2, 1))
        n_cal = min(max(int(0.2 * n), 1), max(n - n_train - 1, 1))
        train_traces = set(traces[:n_train].tolist())
        cal_traces = set(traces[n_train:n_train + n_cal].tolist())
        target_traces = set(traces[n_train + n_cal:].tolist())
    elif protocol == "chronological-holdout":
        if trace_start_map is None:
            raise ValueError("trace_start_map is required for chronological-holdout")
        traces = sorted(unique_traces.tolist(), key=lambda trace: str(trace_start_map[int(trace)]))
        n = len(traces)
        n_target = min(max(int(0.2 * n), 1), max(n - 2, 1))
        source_traces = np.array(traces[: n - n_target], dtype=int)
        target_traces = set(traces[n - n_target :])
        rng = np.random.default_rng(seed)
        rng.shuffle(source_traces)
        split = min(max(int(0.75 * len(source_traces)), 1), max(len(source_traces) - 1, 1))
        train_traces = set(source_traces[:split].tolist())
        cal_traces = set(source_traces[split:].tolist())
    else:
        raise ValueError(f"unknown split protocol: {protocol}")
    return {
        "protocol": protocol,
        "train_idx": np.array([i for i, trace in enumerate(trace_ids) if trace in train_traces]),
        "cal_idx": np.array([i for i, trace in enumerate(trace_ids) if trace in cal_traces]),
        "target_idx": np.array([i for i, trace in enumerate(trace_ids) if trace in target_traces]),
        "train_traces": train_traces,
        "cal_traces": cal_traces,
        "target_traces": target_traces,
    }


def evaluate_irish_protocol(
    features: np.ndarray,
    labels: np.ndarray,
    current_cell_source: np.ndarray,
    trace_ids: np.ndarray,
    speed_col: np.ndarray,
    radio_margin: np.ndarray,
    n_cells: int,
    split: dict,
    alpha: float,
    gamma: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
    trigger_quantile: float,
    replay_config: ReplayConfig,
    delay_grid: list,
    budget_grid: list,
    budget_target: float,
    budget_gamma: float,
    budget_target_grid: list[float],
    budget_gamma_grid: list[float],
    run_seed: int,
):
    set_global_seed(run_seed)
    model = train_model(features, labels, split["train_idx"], split["cal_idx"], n_cells, n_epochs=30)
    cal_probs = get_probs(model, features, split["cal_idx"])
    target_probs = get_probs(model, features, split["target_idx"])
    cal_labels = labels[split["cal_idx"]]
    target_labels = labels[split["target_idx"]]
    cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    current_cells = current_cell_source[split["target_idx"]]
    order = np.lexsort((split["target_idx"], trace_ids[split["target_idx"]]))
    static_threshold = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    raps_threshold = calibrate_raps(cal_probs, cal_labels, alpha=alpha, k_reg=1, lam=0.01, rand=False)
    method_sets = {
        "static-cp": build_static_sets(target_probs, static_threshold),
        "raps-cp": predict_sets_raps(target_probs, raps_threshold, k_reg=1, lam=0.01, rand=False),
    }
    method_sets["triggered-aci"], _, _ = build_triggered_aci_sets(
        target_probs,
        target_labels,
        order,
        cal_scores,
        static_threshold,
        np.max(cal_probs, axis=1),
        alpha,
        gamma,
        trigger_quantile,
        feedback_delay=0,
    )
    method_sets["daci"], _ = build_daci_sets(
        target_probs,
        target_labels,
        order,
        cal_scores,
        alpha,
        daci_gamma_low,
        daci_gamma_high,
        daci_ema_beta,
        feedback_delay=0,
    )
    method_sets["budgeted-daci"], _ = build_budgeted_daci_sets(
        target_probs,
        target_labels,
        order,
        cal_scores,
        alpha,
        daci_gamma_low,
        daci_gamma_high,
        daci_ema_beta,
        0,
        budget_target,
        budget_gamma,
        target_probs.shape[1],
    )
    method_sets["fixed-k"] = build_fixed_k_sets(target_probs, replay_config.k_max)
    method_sets["static-cp-trunc"] = truncate_prediction_sets(method_sets["static-cp"], target_probs, replay_config.k_max)

    def prediction_metrics(prediction_sets: list):
        cp_metrics = evaluate_cp(prediction_sets, target_labels)
        predicted_cell = np.array([
            int(prediction_sets[i][np.argmax(target_probs[i][prediction_sets[i]])])
            for i in range(len(target_labels))
        ])
        ordered_pred = predicted_cell[order]
        ordered_trace = trace_ids[split["target_idx"]][order]
        handover_count = 0
        ping_pong_count = 0
        for trace in np.unique(ordered_trace):
            seq = ordered_pred[ordered_trace == trace]
            handover_count += count_handovers(seq)
            ping_pong_count += count_ping_pong(seq)
        handover_mask = predicted_cell != current_cells
        handover_precision = float(np.mean(predicted_cell[handover_mask] == target_labels[handover_mask])) if np.any(handover_mask) else 0.0
        ho_success = float(np.mean(predicted_cell == target_labels))
        return {
            "coverage": float(cp_metrics["coverage"]),
            "avg_set_size": float(cp_metrics["avg_set_size"]),
            "set_size_std": float(cp_metrics["set_size_std"]),
            "ho_success": ho_success,
            "measurement_load": float(cp_metrics["avg_set_size"] / max(n_cells, 1)),
            "ping_pong_rate": float(ping_pong_count / max(handover_count, 1)),
            "handover_rate": float(np.mean(handover_mask)),
            "handover_precision": handover_precision,
            "wrong_cell_dwell": float(1.0 - ho_success),
            "interruption_proxy": float(1.0 - ho_success),
        }

    method_results = {
        "a3-ttt": {
            "coverage": float(np.mean(current_cells == target_labels)),
            "avg_set_size": 1.0,
            "set_size_std": 0.0,
            "ho_success": float(np.mean(current_cells == target_labels)),
            "measurement_load": float(1.0 / max(n_cells, 1)),
            "ping_pong_rate": 0.0,
            "handover_rate": 0.0,
            "handover_precision": 0.0,
            "wrong_cell_dwell": float(1.0 - np.mean(current_cells == target_labels)),
            "interruption_proxy": float(1.0 - np.mean(current_cells == target_labels)),
        }
    }
    for method, prediction_sets in method_sets.items():
        method_results[method] = prediction_metrics(prediction_sets)

    target_trace_ids = trace_ids[split["target_idx"]]
    method_cover = {
        "a3-ttt": (current_cells == target_labels).astype(float),
        "static-cp": set_cover_array(target_labels, method_sets["static-cp"]),
        "raps-cp": set_cover_array(target_labels, method_sets["raps-cp"]),
        "triggered-aci": set_cover_array(target_labels, method_sets["triggered-aci"]),
        "daci": set_cover_array(target_labels, method_sets["daci"]),
        "budgeted-daci": set_cover_array(target_labels, method_sets["budgeted-daci"]),
        "fixed-k": set_cover_array(target_labels, method_sets["fixed-k"]),
        "static-cp-trunc": set_cover_array(target_labels, method_sets["static-cp-trunc"]),
    }
    method_size = {
        "a3-ttt": np.ones(len(target_labels), dtype=float),
        "static-cp": set_size_array(method_sets["static-cp"]),
        "raps-cp": set_size_array(method_sets["raps-cp"]),
        "triggered-aci": set_size_array(method_sets["triggered-aci"]),
        "daci": set_size_array(method_sets["daci"]),
        "budgeted-daci": set_size_array(method_sets["budgeted-daci"]),
        "fixed-k": set_size_array(method_sets["fixed-k"]),
        "static-cp-trunc": set_size_array(method_sets["static-cp-trunc"]),
    }
    bootstrap_ci = {
        method: trace_bootstrap_ci(target_trace_ids, method_cover[method], method_size[method])
        for method in method_cover
    }
    paired_deltas = {
        "daci_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["daci"],
            method_cover["static-cp"],
            method_size["daci"],
            method_size["static-cp"],
        ),
        "raps_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["raps-cp"],
            method_cover["static-cp"],
            method_size["raps-cp"],
            method_size["static-cp"],
        ),
        "budgeted_minus_daci": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["budgeted-daci"],
            method_cover["daci"],
            method_size["budgeted-daci"],
            method_size["daci"],
        ),
        "budgeted_minus_fixed-k": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["budgeted-daci"],
            method_cover["fixed-k"],
            method_size["budgeted-daci"],
            method_size["fixed-k"],
        ),
        "budgeted_minus_static-cp-trunc": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["budgeted-daci"],
            method_cover["static-cp-trunc"],
            method_size["budgeted-daci"],
            method_size["static-cp-trunc"],
        ),
    }

    budget_sweep = {}
    confidence_values = np.max(target_probs, axis=1)
    for target_size_ratio in budget_target_grid:
        for budget_gamma_value in budget_gamma_grid:
            sweep_sets, _ = build_budgeted_daci_sets(
                target_probs,
                target_labels,
                order,
                cal_scores,
                alpha,
                daci_gamma_low,
                daci_gamma_high,
                daci_ema_beta,
                0,
                target_size_ratio,
                budget_gamma_value,
                target_probs.shape[1],
            )
            sweep_cover = set_cover_array(target_labels, sweep_sets)
            confidence_slice = conditional_summary(
                confidence_values,
                {"budgeted-daci": sweep_cover},
            )
            key = budget_sweep_key(target_size_ratio, budget_gamma_value)
            budget_sweep[key] = {
                "budget_target": float(target_size_ratio),
                "budget_gamma": float(budget_gamma_value),
                **prediction_metrics(sweep_sets),
                "worst_confidence_min": float(
                    min(row["methods"]["budgeted-daci"] for row in confidence_slice["deciles"])
                ),
            }

    delayed = {"triggered-aci": {}, "daci": {}}
    for delay in delay_grid:
        delayed_sets, _, _ = build_triggered_aci_sets(
            target_probs,
            target_labels,
            order,
            cal_scores,
            static_threshold,
            np.max(cal_probs, axis=1),
            alpha,
            gamma,
            trigger_quantile,
            feedback_delay=delay,
        )
        delayed["triggered-aci"][str(delay)] = prediction_metrics(delayed_sets)
        daci_sets, _ = build_daci_sets(
            target_probs,
            target_labels,
            order,
            cal_scores,
            alpha,
            daci_gamma_low,
            daci_gamma_high,
            daci_ema_beta,
            feedback_delay=delay,
        )
        delayed["daci"][str(delay)] = prediction_metrics(daci_sets)

    recalibration = {"source-only": {}, "rolling-window": {}, "warm-start": {}}
    for strategy in recalibration.keys():
        for budget in budget_grid:
            key = budget_label(budget)
            prediction_sets, eval_mask = build_recalibration_sets(
                target_probs,
                target_labels,
                order,
                cal_scores,
                alpha,
                strategy,
                budget,
                0,
            )
            subset_idx = np.where(eval_mask)[0]
            subset_sets = [prediction_sets[i] for i in subset_idx]
            subset_target = target_labels[subset_idx]
            subset_probs = target_probs[subset_idx]
            cp_metrics = evaluate_cp(subset_sets, subset_target)
            predicted_cell = np.array([
                int(subset_sets[i][np.argmax(subset_probs[i][subset_sets[i]])])
                for i in range(len(subset_target))
            ])
            ho_success = float(np.mean(predicted_cell == subset_target))
            recalibration[strategy][key] = {
                "coverage": float(cp_metrics["coverage"]),
                "avg_set_size": float(cp_metrics["avg_set_size"]),
                "ho_success": ho_success,
                "measurement_load": float(cp_metrics["avg_set_size"] / max(n_cells, 1)),
                "wrong_cell_dwell": float(1.0 - ho_success),
                "interruption_proxy": float(1.0 - ho_success),
            }

    cover_arrays = {
        method: set_cover_array(target_labels, prediction_sets)
        for method, prediction_sets in method_sets.items()
    }
    worst_slice = {
        "speed": conditional_summary(speed_col[split["target_idx"]], cover_arrays),
        "confidence": conditional_summary(np.max(target_probs, axis=1), cover_arrays),
        "signal_margin": conditional_summary(radio_margin[split["target_idx"]], cover_arrays),
    }
    return {
        "split_seed": int(run_seed),
        "split": {
            "protocol": split["protocol"],
            "train": int(len(split["train_idx"])),
            "cal": int(len(split["cal_idx"])),
            "target": int(len(split["target_idx"])),
            "source_traces": int(len(split["train_traces"]) + len(split["cal_traces"])),
            "target_traces": int(len(split["target_traces"])),
        },
        "results": method_results,
        "bootstrap_ci": bootstrap_ci,
        "paired_deltas": paired_deltas,
        "budget_sweep": budget_sweep,
        "delayed_feedback": delayed,
        "recalibration": recalibration,
        "worst_slice": worst_slice,
    }


def aggregate_irish_protocol_runs(
    protocol: str,
    runs: list,
    delay_grid: list,
    budget_grid: list,
):
    result_methods = list(runs[0]["results"].keys())
    results = {}
    for method in result_methods:
        results[method] = aggregate_seed_metrics([run["results"][method] for run in runs])

    delayed = {"triggered-aci": {}, "daci": {}}
    for method in delayed.keys():
        for delay in delay_grid:
            delayed[method][str(delay)] = aggregate_seed_metrics(
                [run["delayed_feedback"][method][str(delay)] for run in runs]
            )

    recalibration = {"source-only": {}, "rolling-window": {}, "warm-start": {}}
    for strategy in recalibration.keys():
        for budget in budget_grid:
            key = budget_label(budget)
            recalibration[strategy][key] = aggregate_seed_metrics(
                [run["recalibration"][strategy][key] for run in runs]
            )

    paired_deltas = {}
    for comparison in runs[0]["paired_deltas"].keys():
        paired_deltas[comparison] = aggregate_named_fields(
            [run["paired_deltas"][comparison] for run in runs],
            ["coverage_delta_mean", "avg_set_size_delta_mean"],
        )

    worst_slice_runs = [run["worst_slice"] for run in runs]
    worst_slice = aggregate_conditional_summaries(worst_slice_runs)
    worst_slice_summary = summarize_worst_deciles(worst_slice_runs)
    frontier_methods = pareto_frontier_methods(
        results,
        ["static-cp", "raps-cp", "triggered-aci", "daci", "budgeted-daci", "fixed-k", "static-cp-trunc"],
        y_key="coverage",
        x_key="measurement_load",
    )
    budget_sweep = aggregate_budget_sweep_runs(
        [run["budget_sweep"] for run in runs],
        float(results["daci"]["measurement_load_mean"]),
    )

    split_summary = aggregate_named_fields(
        [run["split"] for run in runs],
        ["train", "cal", "target", "source_traces", "target_traces"],
    )
    split_summary["protocol"] = protocol

    return {
        "protocol": protocol,
        "split_seeds": [int(run["split_seed"]) for run in runs],
        "runs": runs,
        "bootstrap_ci_runs": [
            {"split_seed": int(run["split_seed"]), "bootstrap_ci": run["bootstrap_ci"]}
            for run in runs
        ],
        "split": split_summary,
        "results": results,
        "paired_deltas": paired_deltas,
        "budget_sweep": budget_sweep,
        "delayed_feedback": delayed,
        "recalibration": recalibration,
        "worst_slice": worst_slice,
        "worst_slice_summary": worst_slice_summary,
        "frontier_methods": frontier_methods,
    }


def render_shift_summary_figure(aggregated: dict, figures_dir: Path):
    shifts = ["speed-shift", "shadow-shift", "horizon-shift", "stale-neighbor-shift", "regime-switch"]
    labels = ["Speed", "Shadow", "Horizon", "Stale", "Regime"]
    methods = ["a3-ttt", "static-cp", "raps-cp", "triggered-aci", "daci", "budgeted-daci"]
    x = np.arange(len(shifts))
    width = 0.12
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    for i, method in enumerate(methods):
        vals = [aggregated[shift][method]["ho_success_mean"] for shift in shifts]
        errs = [aggregated[shift][method]["ho_success_std"] for shift in shifts]
        ax.bar(x + (i - (len(methods) - 1) / 2.0) * width, vals, width, yerr=errs, capsize=2, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("HO success")
    ax.set_title("Synthetic hard-shift controller replay")
    ax.legend(ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "controller-shift-summary-v7.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "controller-shift-summary-v7.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_frontier_figure(aggregated: dict, irish_results: dict, figures_dir: Path):
    methods = ["static-cp", "raps-cp", "triggered-aci", "daci", "budgeted-daci", "fixed-k", "static-cp-trunc"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, title, block in [
        (axes[0], "Regime shift", aggregated["regime-switch"]),
        (axes[1], "Irish speed split", irish_results),
    ]:
        for method in methods:
            vals = block[method]
            x = vals["measurement_load_mean"] if "measurement_load_mean" in vals else vals["measurement_load"]
            y = vals["coverage_mean"] if "coverage_mean" in vals else vals["coverage"]
            ax.scatter(x, y, s=42)
            ax.text(x + 0.01, y + 0.003, method, fontsize=7)
        ax.set_xlabel("Measurement load")
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "controller-frontier-v7.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "controller-frontier-v7.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_delayed_feedback_figure(delay_agg: dict, irish_delay: dict, figures_dir: Path):
    delays = sorted(int(key) for key in delay_agg["daci"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, title, block in [
        (axes[0], "Regime shift", delay_agg),
        (axes[1], "Irish speed split", irish_delay),
    ]:
        for method in ["triggered-aci", "daci"]:
            vals = [block[method][str(delay)]["coverage_mean"] if "coverage_mean" in block[method][str(delay)] else block[method][str(delay)]["coverage"] for delay in delays]
            ax.plot(delays, vals, marker="o", linewidth=1.8, label=method)
        ax.set_xlabel("Feedback delay")
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.set_ylim(0.6, 1.02)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "delayed-feedback-v7.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "delayed-feedback-v7.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_recalibration_figure(recalibration: dict, irish_recalibration: dict, figures_dir: Path):
    budget_keys = ["full"] + [str(key) for key in sorted(int(item) for item in recalibration["rolling-window"].keys() if item != "full")]
    x = np.arange(len(budget_keys))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, title, block in [
        (axes[0], "Regime shift", recalibration),
        (axes[1], "Irish speed split", irish_recalibration),
    ]:
        for strategy in ["source-only", "rolling-window", "warm-start"]:
            vals = [block[strategy][budget_key]["coverage_mean"] if "coverage_mean" in block[strategy][budget_key] else block[strategy][budget_key]["coverage"] for budget_key in budget_keys]
            ax.plot(x, vals, marker="o", linewidth=1.8, label=strategy)
        ax.set_xticks(x)
        ax.set_xticklabels(budget_keys)
        ax.set_xlabel("Target calibration budget")
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.set_ylim(0.6, 1.02)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "recalibration-budget-v7.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "recalibration-budget-v7.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_worst_slice_figure(worst_slice: dict, figures_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    panels = [
        ("speed", "Speed decile"),
        ("confidence", "Confidence decile"),
        ("signal_margin", "Signal margin decile"),
    ]
    for ax, (domain, title) in zip(axes, panels):
        for method in ["static-cp", "raps-cp", "daci", "budgeted-daci", "fixed-k", "static-cp-trunc"]:
            vals = [worst_slice[domain][i]["methods"][method]["mean"] for i in range(10)]
            ax.plot(np.arange(1, 11), vals, marker="o", linewidth=1.6, label=method)
        ax.set_title(title)
        ax.set_xlabel("Decile")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.2, 1.02)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "irish-worst-slice-v7.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "irish-worst-slice-v7.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "irish", "all"], default="all")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--n-traj", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--daci-gamma-low", type=float, default=0.005)
    parser.add_argument("--daci-gamma-high", type=float, default=0.02)
    parser.add_argument("--daci-ema-beta", type=float, default=0.95)
    parser.add_argument("--trigger-quantile", type=float, default=0.7)
    parser.add_argument("--delay-grid", type=str, default="0,2,5")
    parser.add_argument("--budget-grid", type=str, default="0,16,32,64")
    parser.add_argument("--budget-target", type=float, default=0.10)
    parser.add_argument("--budget-gamma", type=float, default=0.01)
    parser.add_argument("--budget-target-grid", type=str, default="0.05,0.08,0.10,0.12,0.15")
    parser.add_argument("--budget-gamma-grid", type=str, default="0.01,0.02,0.05")
    parser.add_argument("--a3-offset-db", type=float, default=3.0)
    parser.add_argument("--ttt-steps", type=int, default=2)
    parser.add_argument("--min-dwell-steps", type=int, default=2)
    parser.add_argument("--interruption-margin-db", type=float, default=6.0)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--irish-horizon", type=int, default=10)
    parser.add_argument("--irish-split-seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--output-json", type=str, default="figures/shift-results-v7.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    irish_split_seeds = [int(item.strip()) for item in args.irish_split_seeds.split(",") if item.strip()]
    delay_grid = parse_int_list(args.delay_grid)
    budget_grid = parse_int_list(args.budget_grid)
    budget_target_grid = parse_float_list(args.budget_target_grid)
    budget_gamma_grid = parse_float_list(args.budget_gamma_grid)
    replay_config = ReplayConfig(
        a3_offset_db=args.a3_offset_db,
        ttt_steps=args.ttt_steps,
        min_dwell_steps=args.min_dwell_steps,
        k_max=args.k_max,
        interruption_margin_db=args.interruption_margin_db,
    )

    start = time.time()
    payload = {
        "metadata": {
            "created_at_iso_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "git_commit": get_git_commit(project_dir),
            "argv": sys.argv,
            "mode": args.mode,
            "seeds": seeds,
            "n_traj": args.n_traj,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "daci_gamma_low": args.daci_gamma_low,
            "daci_gamma_high": args.daci_gamma_high,
            "daci_ema_beta": args.daci_ema_beta,
            "trigger_quantile": args.trigger_quantile,
            "delay_grid": delay_grid,
            "budget_grid": budget_grid,
            "budget_target": args.budget_target,
            "budget_gamma": args.budget_gamma,
            "budget_target_grid": budget_target_grid,
            "budget_gamma_grid": budget_gamma_grid,
            "replay_config": {
                "a3_offset_db": args.a3_offset_db,
                "ttt_steps": args.ttt_steps,
                "min_dwell_steps": args.min_dwell_steps,
                "interruption_margin_db": args.interruption_margin_db,
                "k_max": args.k_max,
            },
            "max_files": args.max_files,
            "irish_horizon": args.irish_horizon,
            "irish_split_seeds": irish_split_seeds,
            "device": device,
        },
        "synthetic": {},
        "irish": {},
        "elapsed_seconds": 0.0,
    }

    if args.mode in ["synthetic", "all"]:
        synthetic_seed_results = {shift: {method: [] for method in METHOD_ORDER} for shift in SHIFT_ORDER}
        delay_seed_results = {"regime-switch": {"triggered-aci": {str(delay): [] for delay in delay_grid}, "daci": {str(delay): [] for delay in delay_grid}}}
        recal_seed_results = {"regime-switch": {strategy: {budget_label(budget): [] for budget in budget_grid} for strategy in ["source-only", "rolling-window", "warm-start"]}}
        worst_slice_seed = []

        for seed in seeds:
            seed_out = evaluate_synthetic_seed(
                seed=seed,
                n_traj=args.n_traj,
                epochs=args.epochs,
                batch_size=args.batch_size,
                alpha=args.alpha,
                gamma=args.gamma,
                daci_gamma_low=args.daci_gamma_low,
                daci_gamma_high=args.daci_gamma_high,
                daci_ema_beta=args.daci_ema_beta,
                trigger_quantile=args.trigger_quantile,
                replay_config=replay_config,
                device=device,
                delay_grid=delay_grid,
                budget_grid=budget_grid,
                budget_target=args.budget_target,
                budget_gamma=args.budget_gamma,
            )
            for shift in SHIFT_ORDER:
                for method in METHOD_ORDER:
                    synthetic_seed_results[shift][method].append(seed_out["shift_results"][shift][method])
            for delay in delay_grid:
                for method in ["triggered-aci", "daci"]:
                    delay_seed_results["regime-switch"][method][str(delay)].append(
                        seed_out["delayed_feedback"]["regime-switch"][method][str(delay)]
                    )
            for strategy in recal_seed_results["regime-switch"].keys():
                for budget in budget_grid:
                    key = budget_label(budget)
                    recal_seed_results["regime-switch"][strategy][key].append(
                        seed_out["recalibration"]["regime-switch"][strategy][key]
                    )
            worst_slice_seed.append(seed_out["worst_slice"]["regime-switch"])

        aggregated = {}
        for shift in SHIFT_ORDER:
            aggregated[shift] = {}
            for method in METHOD_ORDER:
                aggregated[shift][method] = aggregate_seed_metrics(synthetic_seed_results[shift][method])

        delayed_agg = {"regime-switch": {"triggered-aci": {}, "daci": {}}}
        for method in ["triggered-aci", "daci"]:
            for delay in delay_grid:
                delayed_agg["regime-switch"][method][str(delay)] = aggregate_seed_metrics(
                    delay_seed_results["regime-switch"][method][str(delay)]
                )

        recalibration_agg = {"regime-switch": {}}
        for strategy in recal_seed_results["regime-switch"].keys():
            recalibration_agg["regime-switch"][strategy] = {}
            for budget in budget_grid:
                key = budget_label(budget)
                recalibration_agg["regime-switch"][strategy][key] = aggregate_seed_metrics(
                    recal_seed_results["regime-switch"][strategy][key]
                )

        paired = {
            "regime-shift": {
                "daci_minus_static_ho_success": paired_delta_with_ci(
                    synthetic_seed_results["regime-switch"]["daci"],
                    synthetic_seed_results["regime-switch"]["static-cp"],
                    "ho_success",
                ),
                "budgeted_minus_daci_measurement_load": paired_delta_with_ci(
                    synthetic_seed_results["regime-switch"]["budgeted-daci"],
                    synthetic_seed_results["regime-switch"]["daci"],
                    "measurement_load",
                ),
            },
            "shadow-shift": {
                "raps_minus_static_ho_success": paired_delta_with_ci(
                    synthetic_seed_results["shadow-shift"]["raps-cp"],
                    synthetic_seed_results["shadow-shift"]["static-cp"],
                    "ho_success",
                ),
            },
        }

        payload["synthetic"] = {
            "aggregated": aggregated,
            "delayed_feedback": delayed_agg,
            "recalibration": recalibration_agg,
            "worst_slice": aggregate_conditional_summaries(worst_slice_seed),
            "paired_deltas": paired,
        }
        render_shift_summary_figure(aggregated, figures_dir)

    if args.mode in ["irish", "all"]:
        data_dir = ensure_irish_data(project_dir)
        df = load_driving_traces(data_dir, max_files=args.max_files)
        data = preprocess_for_handover(df, prediction_horizon=args.irish_horizon)
        features = data["features"]
        labels = data["next_cell"]
        trace_ids = data["trace_id"]
        speed_col = features[:, 2]
        radio_margin = np.abs(features[:, 0] - features[:, 3])
        trace_start_map = df.groupby("trace_id")["Timestamp"].min().astype(str).to_dict()
        protocols = {}
        for protocol in ["speed-split", "trace-holdout", "chronological-holdout"]:
            runs = []
            for split_seed in irish_split_seeds:
                split = build_trace_split(
                    trace_ids,
                    speed_col,
                    split_seed,
                    protocol,
                    trace_start_map=trace_start_map,
                )
                runs.append(
                    evaluate_irish_protocol(
                        features=features,
                        labels=labels,
                        current_cell_source=data["current_cell"],
                        trace_ids=trace_ids,
                        speed_col=speed_col,
                        radio_margin=radio_margin,
                        n_cells=data["n_cells"],
                        split=split,
                        alpha=args.alpha,
                        gamma=args.gamma,
                        daci_gamma_low=args.daci_gamma_low,
                        daci_gamma_high=args.daci_gamma_high,
                        daci_ema_beta=args.daci_ema_beta,
                        trigger_quantile=args.trigger_quantile,
                        replay_config=replay_config,
                        delay_grid=delay_grid,
                        budget_grid=budget_grid,
                        budget_target=args.budget_target,
                        budget_gamma=args.budget_gamma,
                        budget_target_grid=budget_target_grid,
                        budget_gamma_grid=budget_gamma_grid,
                        run_seed=split_seed,
                    )
                )
            protocols[protocol] = aggregate_irish_protocol_runs(protocol, runs, delay_grid, budget_grid)
        payload["irish"] = {"protocols": protocols}

    if payload["synthetic"] and payload["irish"]:
        render_frontier_figure(
            payload["synthetic"]["aggregated"],
            payload["irish"]["protocols"]["speed-split"]["results"],
            figures_dir,
        )
        render_delayed_feedback_figure(
            payload["synthetic"]["delayed_feedback"]["regime-switch"],
            payload["irish"]["protocols"]["speed-split"]["delayed_feedback"],
            figures_dir,
        )
        render_recalibration_figure(
            payload["synthetic"]["recalibration"]["regime-switch"],
            payload["irish"]["protocols"]["speed-split"]["recalibration"],
            figures_dir,
        )
        render_worst_slice_figure(
            payload["irish"]["protocols"]["speed-split"]["worst_slice"],
            figures_dir,
        )

    payload["elapsed_seconds"] = time.time() - start
    out_path = project_dir / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
