import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.handover.baseline import (
    evaluate_3db_baseline,
    simulate_policy_serving_sequence,
    count_handovers,
    count_ping_pong,
)
from src.handover.conformal import (
    calibrate_threshold,
    calibrate_weighted_threshold,
    calibrate_raps,
    predict_sets,
    predict_sets_raps,
    evaluate_cp,
    estimate_density_ratio_weights,
    AdaptiveConformalInference,
)
from src.handover.predictor import train_predictor
from src.handover.synthetic_data import generate_dataset, NetworkConfig, MobilityConfig


@dataclass
class ShiftConfig:
    name: str
    noise_std_db: float
    measurement_noise_db: float
    speed_min: float
    speed_max: float


SHIFT_CONFIGS = {
    "iid": ShiftConfig("IID", 6.0, 4.0, 1.0, 30.0),
    "speed-shift": ShiftConfig("Speed Shift", 6.0, 4.0, 20.0, 50.0),
    "measurement-noise-shift": ShiftConfig("Measurement Noise Shift", 6.0, 8.0, 1.0, 30.0),
    "shadow-shift": ShiftConfig("Shadow Shift", 10.0, 4.0, 1.0, 30.0),
}

SHIFT_ORDER = [
    "iid",
    "speed-shift",
    "measurement-noise-shift",
    "shadow-shift",
    "regime-switch",
]

METHOD_ORDER = ["3db", "top1", "top3", "static-cp", "raps-cp", "aci", "daci", "triggered-aci", "weighted-cp", "ensemble-cp"]


def parse_env_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


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


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if len(values) < window:
        return np.array([values.mean()])
    c = np.cumsum(np.insert(values, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def split_by_trajectory(traj_ids: np.ndarray, n_traj: int):
    train_traj = set(range(int(0.6 * n_traj)))
    cal_traj = set(range(int(0.6 * n_traj), int(0.8 * n_traj)))
    test_traj = set(range(int(0.8 * n_traj), n_traj))
    train_idx = np.array([i for i, t in enumerate(traj_ids) if t in train_traj])
    cal_idx = np.array([i for i, t in enumerate(traj_ids) if t in cal_traj])
    test_idx = np.array([i for i, t in enumerate(traj_ids) if t in test_traj])
    return train_idx, cal_idx, test_idx


def source_stats(data: dict) -> dict:
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


def feature_matrix_for_ratio(data: dict, idx: np.ndarray, stats: dict):
    rsrp, speed = normalize_features(data, idx, stats)
    return np.hstack([rsrp, speed[:, None]])


def simulate_handover_protocol(pred_sets: list, true_labels: np.ndarray, n_cells: int, k_max: int):
    n = len(pred_sets)
    successes = 0
    measurements = 0
    predictive = 0
    rlf = 0
    for i in range(n):
        s = pred_sets[i]
        if len(s) <= k_max:
            predictive += 1
            measurements += len(s)
            if true_labels[i] in s:
                successes += 1
            else:
                rlf += 1
        else:
            measurements += n_cells
            successes += 1
    return {
        "ho_success": successes / n,
        "measurement_overhead": measurements / (n * n_cells),
        "predictive_rate": predictive / n,
        "rlf_proxy": rlf / n,
    }


def sort_relative_by_traj_time(data: dict, idx: np.ndarray):
    traj = data["trajectory_id"][idx]
    step = data["time_step"][idx]
    return np.lexsort((step, traj))


def build_aci_sets(probs: np.ndarray, labels: np.ndarray, order: np.ndarray, cal_scores: np.ndarray, alpha: float, gamma: float):
    aci = AdaptiveConformalInference(alpha=alpha, gamma=gamma)
    sets = [None] * len(labels)
    covered = np.zeros(len(labels), dtype=bool)
    for rel in order:
        threshold = aci.get_threshold(cal_scores)
        s = np.where(probs[rel] >= 1 - threshold)[0]
        if len(s) == 0:
            s = np.array([int(np.argmax(probs[rel]))])
        sets[rel] = s
        c = int(labels[rel] in s)
        covered[rel] = bool(c)
        aci.update(bool(c))
    return sets, covered


def build_daci_sets(
    probs: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float,
    gamma_low: float,
    gamma_high: float,
    ema_beta: float,
):
    alpha_t = float(alpha)
    err_ema = float(alpha)
    sets = [None] * len(labels)
    covered = np.zeros(len(labels), dtype=bool)
    n_cal = len(cal_scores)
    for rel in order:
        q_level = np.ceil((n_cal + 1) * (1 - alpha_t)) / n_cal
        q_level = np.clip(q_level, 0, 1)
        threshold = np.quantile(cal_scores, q_level, method="higher")
        s = np.where(probs[rel] >= 1 - threshold)[0]
        if len(s) == 0:
            s = np.array([int(np.argmax(probs[rel]))])
        sets[rel] = s
        c = int(labels[rel] in s)
        covered[rel] = bool(c)
        err = 1 - c
        err_ema = ema_beta * err_ema + (1 - ema_beta) * err
        gamma_t = gamma_high if err_ema > alpha else gamma_low
        alpha_t = alpha_t + gamma_t * (alpha - err)
        alpha_t = np.clip(alpha_t, 0.001, 0.999)
    return sets, covered


def topk_sets(probs: np.ndarray, k: int):
    idx = np.argsort(probs, axis=1)[:, -k:]
    out = []
    for i in range(len(probs)):
        out.append(np.array(sorted(idx[i].tolist())))
    return out


def set_cover_array(labels: np.ndarray, sets: list):
    return np.array([int(labels[i] in sets[i]) for i in range(len(labels))], dtype=float)


def ordered_cover_list(labels: np.ndarray, sets: list, order: np.ndarray):
    return [int(labels[i] in sets[i]) for i in order]


def ping_pong_rate_for_policy(data: dict, idx: np.ndarray, policy: str, ml_predictions=None, pred_sets=None, k_max=5):
    traj_ids = data["trajectory_id"][idx]
    rsrp = data["rsrp"][idx]
    initial_serving = data["serving_cell"][idx]
    unique = np.unique(traj_ids)
    ho = 0
    pp = 0
    for t in unique:
        mask = traj_ids == t
        seq = simulate_policy_serving_sequence(
            rsrp[mask],
            int(initial_serving[mask][0]),
            policy=policy,
            ml_predictions=None if ml_predictions is None else ml_predictions[mask],
            prediction_sets=None if pred_sets is None else [pred_sets[i] for i in np.where(mask)[0]],
            k_max=k_max,
        )
        ho += count_handovers(seq)
        pp += count_ping_pong(seq)
    return pp / max(ho, 1)


def aggregate_seed_metrics(seed_metrics: list):
    keys = seed_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in seed_metrics], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            half = 1.96 * sem
        else:
            half = 0.0
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std
        out[f"{k}_ci95_low"] = mean - half
        out[f"{k}_ci95_high"] = mean + half
    return out


def paired_delta_with_ci(seed_a: list, seed_b: list, field: str, n_boot: int = 2000):
    a = np.array([x[field] for x in seed_a], dtype=float)
    b = np.array([x[field] for x in seed_b], dtype=float)
    d = a - b
    rng = np.random.default_rng(123)
    boot = []
    n = len(d)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(d[idx].mean())
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5])
    return {
        "mean_delta": float(d.mean()),
        "std_delta": float(d.std()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def generate_shift_data(seed: int, n_traj: int, cfg: ShiftConfig):
    network = NetworkConfig(
        n_gnb_x=4,
        n_gnb_y=4,
        cell_radius=150.0,
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
        prediction_horizon=10,
        measurement_noise_db=cfg.measurement_noise_db,
    )
    return data


def make_regime_switch_data(seed: int, n_traj: int):
    data_a = generate_shift_data(seed, n_traj, SHIFT_CONFIGS["iid"])
    harsh = ShiftConfig("Regime Harsh", 10.0, 8.0, 20.0, 50.0)
    data_b = generate_shift_data(seed + 2000, n_traj, harsh)

    _, _, test_a = split_by_trajectory(data_a["trajectory_id"], n_traj)
    _, _, test_b = split_by_trajectory(data_b["trajectory_id"], n_traj)

    m = min(len(test_a), len(test_b))
    m = m // 2
    idx_a = test_a[:m]
    idx_b = test_b[:m]

    out = {}
    out["rsrp"] = np.concatenate([data_a["rsrp"][idx_a], data_b["rsrp"][idx_b]], axis=0)
    out["serving_cell"] = np.concatenate([data_a["serving_cell"][idx_a], data_b["serving_cell"][idx_b]], axis=0)
    out["optimal_cell"] = np.concatenate([data_a["optimal_cell"][idx_a], data_b["optimal_cell"][idx_b]], axis=0)
    out["ue_speed"] = np.concatenate([data_a["ue_speed"][idx_a], data_b["ue_speed"][idx_b]], axis=0)
    out["trajectory_id"] = np.arange(2 * m)
    out["time_step"] = np.zeros(2 * m, dtype=int)
    out["n_cells"] = int(data_a["n_cells"])
    out["phase_boundary"] = int(m)
    return out


def evaluate_methods_for_shift(
    model,
    source_data,
    source_idx,
    source_cal_probs,
    source_cal_labels,
    source_cal_scores,
    target_data,
    target_idx,
    stats,
    alpha,
    gamma,
    daci_gamma_low,
    daci_gamma_high,
    daci_ema_beta,
    raps_k_reg,
    raps_lam,
    trigger_quantile,
    device,
    k_max,
    ensemble_source_cal_probs=None,
    ensemble_target_probs=None,
):
    probs = probs_with_stats(model, target_data, target_idx, stats, device)
    labels = target_data["optimal_cell"][target_idx]

    top1_pred = np.argmax(probs, axis=1)
    top1_sets = [np.array([int(v)]) for v in top1_pred]
    top3 = topk_sets(probs, 3)

    threshold = calibrate_threshold(source_cal_probs, source_cal_labels, alpha=alpha)
    sets_static = predict_sets(probs, threshold)
    threshold_raps = calibrate_raps(
        source_cal_probs,
        source_cal_labels,
        alpha=alpha,
        k_reg=raps_k_reg,
        lam=raps_lam,
        rand=False,
    )
    sets_raps = predict_sets_raps(
        probs,
        threshold_raps,
        k_reg=raps_k_reg,
        lam=raps_lam,
        rand=False,
    )

    source_train_features = feature_matrix_for_ratio(source_data, source_idx["train"], stats)
    source_cal_features = feature_matrix_for_ratio(source_data, source_idx["cal"], stats)
    target_features = feature_matrix_for_ratio(target_data, target_idx, stats)
    weights = estimate_density_ratio_weights(source_train_features, target_features, source_cal_features)
    threshold_weighted = calibrate_weighted_threshold(source_cal_probs, source_cal_labels, weights, alpha=alpha)
    sets_weighted = predict_sets(probs, threshold_weighted)

    order = sort_relative_by_traj_time(target_data, target_idx)
    sets_aci, covered_aci = build_aci_sets(probs, labels, order, source_cal_scores, alpha=alpha, gamma=gamma)
    sets_daci, covered_daci = build_daci_sets(
        probs,
        labels,
        order,
        source_cal_scores,
        alpha=alpha,
        gamma_low=daci_gamma_low,
        gamma_high=daci_gamma_high,
        ema_beta=daci_ema_beta,
    )
    source_conf = np.max(source_cal_probs, axis=1)
    trigger_tau = float(np.quantile(source_conf, trigger_quantile))
    target_conf = np.max(probs, axis=1)
    sets_triggered = []
    covered_triggered = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        use_aci = target_conf[i] < trigger_tau
        s = sets_aci[i] if use_aci else sets_static[i]
        sets_triggered.append(s)
        covered_triggered[i] = bool(labels[i] in s)

    sets_ensemble = None
    if ensemble_source_cal_probs is not None and ensemble_target_probs is not None:
        threshold_ensemble = calibrate_threshold(ensemble_source_cal_probs, source_cal_labels, alpha=alpha)
        sets_ensemble = predict_sets(ensemble_target_probs, threshold_ensemble)

    baseline = evaluate_3db_baseline(target_data["rsrp"][target_idx], target_data["serving_cell"][target_idx], labels)

    out = {}

    method_defs = [
        ("top1", top1_sets, "ml_top1"),
        ("top3", top3, "cp_adaptive"),
        ("static-cp", sets_static, "cp_adaptive"),
        ("raps-cp", sets_raps, "cp_adaptive"),
        ("aci", sets_aci, "cp_adaptive"),
        ("daci", sets_daci, "cp_adaptive"),
        ("triggered-aci", sets_triggered, "cp_adaptive"),
        ("weighted-cp", sets_weighted, "cp_adaptive"),
    ]
    if sets_ensemble is not None:
        method_defs.append(("ensemble-cp", sets_ensemble, "cp_adaptive"))

    method_cover = {}
    for method, sets, policy in method_defs:
        ev = evaluate_cp(sets, labels)
        ho = simulate_handover_protocol(sets, labels, int(target_data["n_cells"]), k_max=k_max)
        pp = ping_pong_rate_for_policy(target_data, target_idx, policy, ml_predictions=top1_pred, pred_sets=sets, k_max=k_max)
        cover = set_cover_array(labels, sets)
        method_cover[method] = cover
        out[method] = {
            "coverage": float(ev["coverage"]),
            "avg_set_size": float(ev["avg_set_size"]),
            "ho_success": float(ho["ho_success"]),
            "measurement_overhead": float(ho["measurement_overhead"]),
            "rlf_proxy": float(ho["rlf_proxy"]),
            "ping_pong_rate": float(pp),
        }

    out["3db"] = {
        "coverage": float(baseline["accuracy"]),
        "avg_set_size": 1.0,
        "ho_success": float(baseline["accuracy"]),
        "measurement_overhead": float(1.0 / int(target_data["n_cells"])),
        "rlf_proxy": float(1 - baseline["accuracy"]),
        "ping_pong_rate": float(ping_pong_rate_for_policy(target_data, target_idx, "3db")),
    }

    rolling = {
        "static-cp": ordered_cover_list(labels, sets_static, order),
        "raps-cp": ordered_cover_list(labels, sets_raps, order),
        "aci": [int(covered_aci[i]) for i in order],
        "daci": [int(covered_daci[i]) for i in order],
        "triggered-aci": [int(covered_triggered[i]) for i in order],
        "weighted-cp": ordered_cover_list(labels, sets_weighted, order),
    }
    if sets_ensemble is not None:
        rolling["ensemble-cp"] = ordered_cover_list(labels, sets_ensemble, order)

    conditional = {
        "speed_deciles": conditional_coverage_summary(target_data["ue_speed"][target_idx].astype(float), method_cover),
        "confidence_deciles": conditional_coverage_summary(target_conf.astype(float), method_cover),
    }

    cache = {
        "probs": probs,
        "labels": labels,
        "order": order,
        "top1_pred": top1_pred,
        "method_cover": method_cover,
        "conditional": conditional,
    }

    return out, rolling, cache


def render_shift_figures(aggregated: dict, rolling: dict, figures_dir: Path, alpha: float):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    })

    shift_labels = ["iid", "speed-shift", "measurement-noise-shift", "shadow-shift"]
    x = np.arange(len(shift_labels))
    methods = ["top1", "static-cp", "raps-cp", "aci", "weighted-cp"]
    if "coverage_mean" in aggregated["shadow-shift"].get("ensemble-cp", {}):
        methods.append("ensemble-cp")
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for i, method in enumerate(methods):
        means = [aggregated[s][method]["coverage_mean"] for s in shift_labels]
        errs = [aggregated[s][method]["coverage_std"] for s in shift_labels]
        ax.bar(x + (i - (len(methods) - 1) / 2.0) * width, means, width, yerr=errs, capsize=2, label=method)
    ax.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.5, label="target")
    ax.set_xticks(x)
    ax.set_xticklabels(["IID", "Speed", "MeasNoise", "Shadow"])
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage Under Distribution Shift")
    ax.set_ylim(0.45, 1.02)
    ax.legend(ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "shift-coverage-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "shift-coverage-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    methods = ["static-cp", "raps-cp", "aci", "daci", "weighted-cp"]
    if "ensemble-cp" in rolling:
        methods.append("ensemble-cp")
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for method in methods:
        vals = np.array(rolling[method], dtype=float)
        ax.plot(vals, label=method, linewidth=1.8)
    ax.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Rolling coverage")
    ax.set_title("Regime-Switch Rolling Coverage (window=200)")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "regime-switch-rolling-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "regime-switch-rolling-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_daci_combo_key(key: str):
    vals = {}
    for item in key.split("|"):
        k, v = item.split("=", 1)
        vals[k] = float(v)
    return vals


def make_decile_bins(values: np.ndarray):
    n = len(values)
    order = np.argsort(values)
    bins = np.zeros(n, dtype=int)
    bins[order] = (np.arange(n) * 10) // max(n, 1)
    bins = np.clip(bins, 0, 9)
    return bins


def conditional_coverage_summary(values: np.ndarray, method_cover: dict):
    bins = make_decile_bins(values)
    out = {"deciles": []}
    methods = list(method_cover.keys())
    for decile in range(10):
        idx = np.where(bins == decile)[0]
        item = {
            "decile": decile,
            "n": int(len(idx)),
            "methods": {},
        }
        for method in methods:
            if len(idx) == 0:
                item["methods"][method] = float("nan")
            else:
                item["methods"][method] = float(np.mean(method_cover[method][idx]))
        out["deciles"].append(item)
    return out


def aggregate_conditional_seed_summaries(seed_summaries: list):
    out = {}
    for domain in ["speed_deciles", "confidence_deciles"]:
        out[domain] = []
        methods = list(seed_summaries[0][domain]["deciles"][0]["methods"].keys())
        for decile in range(10):
            n_vals = [s[domain]["deciles"][decile]["n"] for s in seed_summaries]
            row = {
                "decile": decile,
                "n_mean": float(np.mean(n_vals)),
                "methods": {},
            }
            for method in methods:
                vals = [s[domain]["deciles"][decile]["methods"][method] for s in seed_summaries]
                row["methods"][method] = {
                    "coverage_mean": float(np.mean(vals)),
                    "coverage_std": float(np.std(vals)),
                }
            out[domain].append(row)
    return out


def evaluate_aci_gamma_grid(target_data: dict, target_idx: np.ndarray, cache: dict, cal_scores: np.ndarray, alpha: float, gammas: list, k_max: int):
    out = {}
    probs = cache["probs"]
    labels = cache["labels"]
    order = cache["order"]
    top1_pred = cache["top1_pred"]

    for gamma in gammas:
        sets_aci, covered_aci = build_aci_sets(probs, labels, order, cal_scores, alpha=alpha, gamma=gamma)
        ev = evaluate_cp(sets_aci, labels)
        ho = simulate_handover_protocol(sets_aci, labels, int(target_data["n_cells"]), k_max=k_max)
        pp = ping_pong_rate_for_policy(target_data, target_idx, "cp_adaptive", ml_predictions=top1_pred, pred_sets=sets_aci, k_max=k_max)
        out[str(gamma)] = {
            "coverage": float(ev["coverage"]),
            "avg_set_size": float(ev["avg_set_size"]),
            "ho_success": float(ho["ho_success"]),
            "measurement_overhead": float(ho["measurement_overhead"]),
            "rlf_proxy": float(ho["rlf_proxy"]),
            "ping_pong_rate": float(pp),
            "rolling_coverage": [int(covered_aci[i]) for i in order],
        }
    return out


def evaluate_trigger_quantile_grid(
    target_data: dict,
    target_idx: np.ndarray,
    cache: dict,
    cal_scores: np.ndarray,
    source_cal_probs: np.ndarray,
    source_cal_labels: np.ndarray,
    alpha: float,
    gamma: float,
    quantiles: list,
    k_max: int,
):
    out = {}
    probs = cache["probs"]
    labels = cache["labels"]
    top1_pred = cache["top1_pred"]
    threshold = calibrate_threshold(source_cal_probs, source_cal_labels, alpha=alpha)
    sets_static = predict_sets(probs, threshold)
    order = cache["order"]
    sets_aci, _ = build_aci_sets(probs, labels, order, cal_scores, alpha=alpha, gamma=gamma)
    source_conf = np.max(source_cal_probs, axis=1)
    target_conf = np.max(probs, axis=1)

    for q in quantiles:
        tau = float(np.quantile(source_conf, q))
        sets_triggered = []
        for i in range(len(labels)):
            s = sets_aci[i] if target_conf[i] < tau else sets_static[i]
            sets_triggered.append(s)
        ev = evaluate_cp(sets_triggered, labels)
        ho = simulate_handover_protocol(sets_triggered, labels, int(target_data["n_cells"]), k_max=k_max)
        pp = ping_pong_rate_for_policy(target_data, target_idx, "cp_adaptive", ml_predictions=top1_pred, pred_sets=sets_triggered, k_max=k_max)
        out[str(q)] = {
            "coverage": float(ev["coverage"]),
            "avg_set_size": float(ev["avg_set_size"]),
            "ho_success": float(ho["ho_success"]),
            "measurement_overhead": float(ho["measurement_overhead"]),
            "rlf_proxy": float(ho["rlf_proxy"]),
            "ping_pong_rate": float(pp),
            "trigger_threshold": tau,
        }
    return out


def evaluate_daci_robustness_grid(
    target_data: dict,
    target_idx: np.ndarray,
    cache: dict,
    cal_scores: np.ndarray,
    alpha: float,
    gamma_low_grid: list,
    gamma_high_grid: list,
    ema_beta_grid: list,
    k_max: int,
):
    out = {}
    probs = cache["probs"]
    labels = cache["labels"]
    order = cache["order"]
    top1_pred = cache["top1_pred"]
    for gamma_low in gamma_low_grid:
        for gamma_high in gamma_high_grid:
            if gamma_low > gamma_high:
                continue
            for ema_beta in ema_beta_grid:
                sets_daci, covered_daci = build_daci_sets(
                    probs,
                    labels,
                    order,
                    cal_scores,
                    alpha=alpha,
                    gamma_low=gamma_low,
                    gamma_high=gamma_high,
                    ema_beta=ema_beta,
                )
                ev = evaluate_cp(sets_daci, labels)
                ho = simulate_handover_protocol(sets_daci, labels, int(target_data["n_cells"]), k_max=k_max)
                pp = ping_pong_rate_for_policy(target_data, target_idx, "cp_adaptive", ml_predictions=top1_pred, pred_sets=sets_daci, k_max=k_max)
                key = f"gamma_low={gamma_low}|gamma_high={gamma_high}|ema_beta={ema_beta}"
                out[key] = {
                    "coverage": float(ev["coverage"]),
                    "avg_set_size": float(ev["avg_set_size"]),
                    "ho_success": float(ho["ho_success"]),
                    "measurement_overhead": float(ho["measurement_overhead"]),
                    "rlf_proxy": float(ho["rlf_proxy"]),
                    "ping_pong_rate": float(pp),
                    "rolling_coverage": [int(covered_daci[i]) for i in order],
                }
    return out


def render_aci_gamma_figure(gamma_agg: dict, figures_dir: Path, alpha: float):
    gammas = sorted([float(k) for k in gamma_agg.keys()])
    cov = [gamma_agg[str(g)]["coverage_mean"] for g in gammas]
    cov_err = [gamma_agg[str(g)]["coverage_std"] for g in gammas]
    sz = [gamma_agg[str(g)]["avg_set_size_mean"] for g in gammas]
    sz_err = [gamma_agg[str(g)]["avg_set_size_std"] for g in gammas]

    fig, ax1 = plt.subplots(figsize=(8.5, 3.7))
    ax1.errorbar(gammas, cov, yerr=cov_err, marker="o", linewidth=1.8, capsize=2, label="coverage")
    ax1.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("ACI gamma")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0.75, 0.93)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.errorbar(gammas, sz, yerr=sz_err, marker="s", linewidth=1.6, capsize=2, color="black", label="set size")
    ax2.set_ylabel("Avg set size")
    ax2.set_ylim(3.0, 6.8)

    ax1.set_title("Regime-Switch ACI Gamma Tradeoff")
    fig.tight_layout()
    fig.savefig(figures_dir / "aci-gamma-ablation-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "aci-gamma-ablation-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_trigger_quantile_figure(trigger_agg: dict, figures_dir: Path, alpha: float):
    quantiles = sorted([float(k) for k in trigger_agg.keys()])
    cov = [trigger_agg[str(q)]["coverage_mean"] for q in quantiles]
    cov_err = [trigger_agg[str(q)]["coverage_std"] for q in quantiles]
    overhead = [trigger_agg[str(q)]["measurement_overhead_mean"] for q in quantiles]
    overhead_err = [trigger_agg[str(q)]["measurement_overhead_std"] for q in quantiles]

    fig, ax1 = plt.subplots(figsize=(8.5, 3.7))
    ax1.errorbar(quantiles, cov, yerr=cov_err, marker="o", linewidth=1.8, capsize=2, label="coverage")
    ax1.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Trigger quantile")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0.80, 0.92)

    ax2 = ax1.twinx()
    ax2.errorbar(quantiles, overhead, yerr=overhead_err, marker="s", linewidth=1.6, capsize=2, color="black", label="overhead")
    ax2.set_ylabel("Measurement overhead")
    ax2.set_ylim(0.28, 0.46)

    ax1.set_title("Regime-Switch Triggered-ACI Quantile Tradeoff")
    fig.tight_layout()
    fig.savefig(figures_dir / "trigger-quantile-ablation-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "trigger-quantile-ablation-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_daci_robustness_figure(daci_robustness: dict, figures_dir: Path):
    shifts = ["shadow-shift", "regime-switch"]
    titles = {"shadow-shift": "Shadow", "regime-switch": "Regime"}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for ax, shift in zip(axes, shifts):
        keys = sorted(daci_robustness[shift].keys())
        xs = [daci_robustness[shift][k]["measurement_overhead_mean"] for k in keys]
        ys = [daci_robustness[shift][k]["coverage_mean"] for k in keys]
        ax.scatter(xs, ys, s=24)
        best_key = max(keys, key=lambda k: daci_robustness[shift][k]["coverage_mean"])
        bx = daci_robustness[shift][best_key]["measurement_overhead_mean"]
        by = daci_robustness[shift][best_key]["coverage_mean"]
        ax.scatter([bx], [by], s=50, marker="*", color="black")
        ax.set_title(f"DACI Grid ({titles[shift]})")
        ax.set_xlabel("Measurement overhead")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.84, 0.94)
    fig.tight_layout()
    fig.savefig(figures_dir / "daci-robustness-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "daci-robustness-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_conditional_coverage_figure(conditional: dict, figures_dir: Path, domain: str, out_name: str, title: str):
    methods = list(conditional[0]["methods"].keys())
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for method in methods:
        y = [conditional[i]["methods"][method]["coverage_mean"] for i in range(10)]
        ax.plot(x, y, marker="o", linewidth=1.6, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(f"{domain} decile")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.6, 0.97)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{out_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_ensemble_vs_cp_figure(aggregated: dict, irish_results: dict, figures_dir: Path):
    methods = ["static-cp", "raps-cp", "aci", "daci", "ensemble-cp"]
    labels = ["Shadow", "Regime", "Irish"]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for i, method in enumerate(methods):
        vals = [
            aggregated["shadow-shift"][method]["coverage_mean"],
            aggregated["regime-switch"][method]["coverage_mean"],
            irish_results[method]["coverage"],
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.65, 0.96)
    ax.set_ylabel("Coverage")
    ax.set_title("Comparator Coverage Across Hard Shifts")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "ensemble-vs-cp-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "ensemble-vs-cp-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_hard_shift_pareto(aggregated: dict, figures_dir: Path):
    method_order = ["static-cp", "raps-cp", "weighted-cp", "triggered-aci", "aci", "daci", "ensemble-cp"]
    shifts = ["shadow-shift", "regime-switch"]
    label_map = {
        "static-cp": "static",
        "raps-cp": "raps",
        "weighted-cp": "weighted",
        "triggered-aci": "triggered",
        "aci": "aci",
        "daci": "daci",
        "ensemble-cp": "ensemble",
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for ax, shift in zip(axes, shifts):
        pts = []
        for method in method_order:
            if "coverage_mean" not in aggregated[shift].get(method, {}):
                continue
            vals = aggregated[shift][method]
            x = vals["measurement_overhead_mean"]
            y = vals["coverage_mean"]
            pts.append((method, x, y))
            ax.scatter(x, y, s=35)
            ax.text(x + 0.003, y + 0.0015, label_map[method], fontsize=7)

        frontier = []
        for method, x, y in pts:
            dominated = False
            for m2, x2, y2 in pts:
                if m2 == method:
                    continue
                if x2 <= x and y2 >= y and (x2 < x or y2 > y):
                    dominated = True
                    break
            if not dominated:
                frontier.append((x, y))
        frontier = sorted(frontier, key=lambda v: v[0])
        fx = [v[0] for v in frontier]
        fy = [v[1] for v in frontier]
        ax.plot(fx, fy, linewidth=1.2, linestyle="--")
        ax.set_title("Shadow" if shift == "shadow-shift" else "Regime")
        ax.set_xlabel("Measurement overhead")
        ax.set_ylabel("Coverage")
        ax.set_xlim(0.15, 0.68)
        ax.set_ylim(0.66, 0.93)

    fig.tight_layout()
    fig.savefig(figures_dir / "hard-shift-pareto-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "hard-shift-pareto-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_irish_shift_script(
    project_dir: Path,
    output_json: Path,
    trigger_quantile: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
    daci_gamma_low_grid: str,
    daci_gamma_high_grid: str,
    daci_ema_beta_grid: str,
    raps_k_reg: int,
    raps_lam: float,
    split_protocol: str,
    extra_split_protocol: str,
    run_ensemble_baseline: bool,
    ensemble_members: int,
    seed: int,
):
    cmd = [
        sys.executable,
        "run-irish-shift-experiment.py",
        "--output-json",
        str(output_json),
        "--trigger-quantile",
        str(trigger_quantile),
        "--daci-gamma-low",
        str(daci_gamma_low),
        "--daci-gamma-high",
        str(daci_gamma_high),
        "--daci-ema-beta",
        str(daci_ema_beta),
        "--daci-gamma-low-grid",
        daci_gamma_low_grid,
        "--daci-gamma-high-grid",
        daci_gamma_high_grid,
        "--daci-ema-beta-grid",
        daci_ema_beta_grid,
        "--raps-k-reg",
        str(raps_k_reg),
        "--raps-lam",
        str(raps_lam),
        "--split-protocol",
        split_protocol,
        "--extra-split-protocol",
        extra_split_protocol,
        "--ensemble-members",
        str(ensemble_members),
        "--seed",
        str(seed),
    ]
    if run_ensemble_baseline:
        cmd.append("--run-ensemble-baseline")
    return subprocess.run(cmd, cwd=project_dir, check=True)


def maybe_run_vast_overflow(use_vast: bool, spent: float, soft_stop: float, hard_cap: float, env_path: Path):
    if not use_vast:
        return {"enabled": False, "attempted": False, "reason": "use-vast=false"}
    if spent >= soft_stop:
        return {"enabled": True, "attempted": False, "reason": "soft-stop-reached"}
    env_vals = parse_env_file(env_path)
    key = env_vals.get("VASTA_API_KEY") or os.environ.get("VASTA_API_KEY")
    if not key:
        return {"enabled": True, "attempted": False, "reason": "missing-key"}
    if hard_cap <= spent:
        return {"enabled": True, "attempted": False, "reason": "hard-cap-reached"}
    return {"enabled": True, "attempted": False, "reason": "local-first-no-overflow-needed"}


def compute_ensemble_probs_for_seed(
    source_data: dict,
    source_idx: dict,
    stats: dict,
    shift_inputs: dict,
    ensemble_members: int,
    seed: int,
    n_epochs: int,
    batch_size: int,
    device: str,
):
    source_cal_stack = []
    target_stack = {shift: [] for shift in shift_inputs.keys()}
    for member in range(ensemble_members):
        set_global_seed(seed * 1000 + member)
        model_member = train_predictor(
            source_data,
            source_idx["train"],
            source_idx["cal"],
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
        source_cal_probs = probs_with_stats(model_member, source_data, source_idx["cal"], stats, device)
        source_cal_stack.append(source_cal_probs)
        for shift, item in shift_inputs.items():
            target_probs = probs_with_stats(model_member, item["target_data"], item["target_idx"], stats, device)
            target_stack[shift].append(target_probs)
    source_cal_mean = np.mean(np.stack(source_cal_stack, axis=0), axis=0)
    target_mean = {}
    for shift, probs_list in target_stack.items():
        target_mean[shift] = np.mean(np.stack(probs_list, axis=0), axis=0)
    return source_cal_mean, target_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic-shift", "irish-shift", "all"], default="all")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1011")
    parser.add_argument("--n-traj", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--daci-gamma-low", type=float, default=0.005)
    parser.add_argument("--daci-gamma-high", type=float, default=0.02)
    parser.add_argument("--daci-ema-beta", type=float, default=0.95)
    parser.add_argument("--daci-gamma-low-grid", type=str, default="0.002,0.005,0.01")
    parser.add_argument("--daci-gamma-high-grid", type=str, default="0.01,0.02,0.05")
    parser.add_argument("--daci-ema-beta-grid", type=str, default="0.9,0.95,0.98")
    parser.add_argument("--raps-k-reg", type=int, default=1)
    parser.add_argument("--raps-lam", type=float, default=0.01)
    parser.add_argument("--trigger-quantile", type=float, default=0.7)
    parser.add_argument("--aci-gamma-grid", type=str, default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--trigger-quantile-grid", type=str, default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--run-ensemble-baseline", action="store_true")
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--irish-seed", type=int, default=42)
    parser.add_argument("--irish-split-protocol", choices=["speed-split", "trace-holdout"], default="speed-split")
    parser.add_argument("--irish-extra-split-protocol", choices=["none", "trace-holdout"], default="trace-holdout")
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--use-vast", action="store_true")
    parser.add_argument("--budget-cap-usd", type=float, default=12.0)
    parser.add_argument("--cost-soft-stop-usd", type=float, default=10.5)
    parser.add_argument("--output-json", type=str, default="figures/shift-results-v6.json")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    gamma_grid = parse_float_list(args.aci_gamma_grid)
    daci_gamma_low_grid = parse_float_list(args.daci_gamma_low_grid)
    daci_gamma_high_grid = parse_float_list(args.daci_gamma_high_grid)
    daci_ema_beta_grid = parse_float_list(args.daci_ema_beta_grid)
    trigger_quantile_grid = parse_float_list(args.trigger_quantile_grid)
    if args.trigger_quantile < 0.0 or args.trigger_quantile > 1.0:
        raise ValueError("--trigger-quantile must be in [0,1]")
    if not trigger_quantile_grid:
        raise ValueError("--trigger-quantile-grid must contain at least one value")
    for q in trigger_quantile_grid:
        if q < 0.0 or q > 1.0:
            raise ValueError("--trigger-quantile-grid values must be in [0,1]")
    if args.daci_gamma_low <= 0.0 or args.daci_gamma_high <= 0.0:
        raise ValueError("--daci-gamma-low and --daci-gamma-high must be > 0")
    if args.daci_gamma_low > args.daci_gamma_high:
        raise ValueError("--daci-gamma-low must be <= --daci-gamma-high")
    if args.daci_ema_beta < 0.0 or args.daci_ema_beta >= 1.0:
        raise ValueError("--daci-ema-beta must be in [0,1)")
    if args.ensemble_members < 1:
        raise ValueError("--ensemble-members must be >= 1")
    if args.raps_k_reg < 1:
        raise ValueError("--raps-k-reg must be >= 1")
    if args.raps_lam < 0.0:
        raise ValueError("--raps-lam must be >= 0")
    if not daci_gamma_low_grid or not daci_gamma_high_grid or not daci_ema_beta_grid:
        raise ValueError("daci grids must be non-empty")
    for val in daci_gamma_low_grid + daci_gamma_high_grid:
        if val <= 0.0:
            raise ValueError("daci gamma grid values must be > 0")
    for val in daci_ema_beta_grid:
        if val < 0.0 or val >= 1.0:
            raise ValueError("daci ema beta grid values must be in [0,1)")

    start = time.time()
    spent_usd = 0.0

    payload = {
        "metadata": {
            "created_at_unix": int(time.time()),
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
            "daci_gamma_low_grid": daci_gamma_low_grid,
            "daci_gamma_high_grid": daci_gamma_high_grid,
            "daci_ema_beta_grid": daci_ema_beta_grid,
            "raps_k_reg": args.raps_k_reg,
            "raps_lam": args.raps_lam,
            "trigger_quantile": args.trigger_quantile,
            "trigger_quantile_grid": trigger_quantile_grid,
            "run_ensemble_baseline": bool(args.run_ensemble_baseline),
            "ensemble_members": args.ensemble_members,
            "irish_seed": args.irish_seed,
            "irish_split_protocol": args.irish_split_protocol,
            "irish_extra_split_protocol": args.irish_extra_split_protocol,
            "k_max": args.k_max,
            "device": device,
            "budget_cap_usd": args.budget_cap_usd,
            "cost_soft_stop_usd": args.cost_soft_stop_usd,
            "strategy": "local-first",
            "created_at_iso_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "git_commit": get_git_commit(project_dir),
            "argv": sys.argv,
        },
        "synthetic_shift": {},
        "irish_shift": {},
        "cost_tracking": {},
    }

    if args.mode in ["synthetic-shift", "all"]:
        aggregated = {s: {m: {} for m in METHOD_ORDER} for s in SHIFT_ORDER}
        regime_rolling_seed = {"static-cp": [], "raps-cp": [], "aci": [], "daci": [], "weighted-cp": []}
        if args.run_ensemble_baseline:
            regime_rolling_seed["ensemble-cp"] = []
        gamma_seed = {str(g): [] for g in gamma_grid}
        trigger_seed = {str(q): [] for q in trigger_quantile_grid}
        daci_robust_seed = {"shadow-shift": {}, "regime-switch": {}}
        conditional_seed = []

        for seed in seeds:
            set_global_seed(seed)
            source_data = generate_shift_data(seed, args.n_traj, SHIFT_CONFIGS["iid"])
            train_idx, cal_idx, _ = split_by_trajectory(source_data["trajectory_id"], args.n_traj)
            source_idx = {"train": train_idx, "cal": cal_idx}
            shift_inputs = {}
            for shift in SHIFT_ORDER:
                if shift == "regime-switch":
                    target_data = make_regime_switch_data(seed, args.n_traj)
                    target_idx = np.arange(len(target_data["optimal_cell"]))
                else:
                    target_data = generate_shift_data(seed + 1000 + SHIFT_ORDER.index(shift), args.n_traj, SHIFT_CONFIGS[shift])
                    _, _, test_idx = split_by_trajectory(target_data["trajectory_id"], args.n_traj)
                    target_idx = test_idx
                shift_inputs[shift] = {"target_data": target_data, "target_idx": target_idx}

            model = train_predictor(
                source_data,
                train_idx,
                cal_idx,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
            )

            stats = source_stats(source_data)
            source_cal_probs = probs_with_stats(model, source_data, cal_idx, stats, device)
            source_cal_labels = source_data["optimal_cell"][cal_idx]
            source_cal_scores = 1 - source_cal_probs[np.arange(len(source_cal_labels)), source_cal_labels]
            ensemble_source_cal_probs = None
            ensemble_target_probs = {}
            if args.run_ensemble_baseline:
                ensemble_source_cal_probs, ensemble_target_probs = compute_ensemble_probs_for_seed(
                    source_data,
                    source_idx,
                    stats,
                    shift_inputs,
                    args.ensemble_members,
                    seed,
                    args.epochs,
                    args.batch_size,
                    device,
                )

            seed_shift_results = {}

            for shift in SHIFT_ORDER:
                target_data = shift_inputs[shift]["target_data"]
                target_idx = shift_inputs[shift]["target_idx"]

                methods_out, rolling, cache = evaluate_methods_for_shift(
                    model,
                    source_data,
                    source_idx,
                    source_cal_probs,
                    source_cal_labels,
                    source_cal_scores,
                    target_data,
                    target_idx,
                    stats,
                    args.alpha,
                    args.gamma,
                    args.daci_gamma_low,
                    args.daci_gamma_high,
                    args.daci_ema_beta,
                    args.raps_k_reg,
                    args.raps_lam,
                    args.trigger_quantile,
                    device,
                    args.k_max,
                    ensemble_source_cal_probs=ensemble_source_cal_probs,
                    ensemble_target_probs=ensemble_target_probs.get(shift),
                )
                seed_shift_results[shift] = methods_out

                if shift == "regime-switch":
                    for method in regime_rolling_seed:
                        if method in rolling:
                            regime_rolling_seed[method].append(rolling[method])
                    gamma_out = evaluate_aci_gamma_grid(
                        target_data,
                        target_idx,
                        cache,
                        source_cal_scores,
                        args.alpha,
                        gamma_grid,
                        args.k_max,
                    )
                    for g in gamma_grid:
                        gamma_seed[str(g)].append({
                            "coverage": gamma_out[str(g)]["coverage"],
                            "avg_set_size": gamma_out[str(g)]["avg_set_size"],
                            "rlf_proxy": gamma_out[str(g)]["rlf_proxy"],
                            "measurement_overhead": gamma_out[str(g)]["measurement_overhead"],
                            "ping_pong_rate": gamma_out[str(g)]["ping_pong_rate"],
                        })
                    trigger_out = evaluate_trigger_quantile_grid(
                        target_data,
                        target_idx,
                        cache,
                        source_cal_scores,
                        source_cal_probs,
                        source_cal_labels,
                        args.alpha,
                        args.gamma,
                        trigger_quantile_grid,
                        args.k_max,
                    )
                    for q in trigger_quantile_grid:
                        trigger_seed[str(q)].append({
                            "coverage": trigger_out[str(q)]["coverage"],
                            "avg_set_size": trigger_out[str(q)]["avg_set_size"],
                            "rlf_proxy": trigger_out[str(q)]["rlf_proxy"],
                            "measurement_overhead": trigger_out[str(q)]["measurement_overhead"],
                            "ping_pong_rate": trigger_out[str(q)]["ping_pong_rate"],
                        })
                    conditional_seed.append(cache["conditional"])
                    daci_out = evaluate_daci_robustness_grid(
                        target_data,
                        target_idx,
                        cache,
                        source_cal_scores,
                        args.alpha,
                        daci_gamma_low_grid,
                        daci_gamma_high_grid,
                        daci_ema_beta_grid,
                        args.k_max,
                    )
                    for key, vals in daci_out.items():
                        if key not in daci_robust_seed["regime-switch"]:
                            daci_robust_seed["regime-switch"][key] = []
                        daci_robust_seed["regime-switch"][key].append({
                            "coverage": vals["coverage"],
                            "avg_set_size": vals["avg_set_size"],
                            "rlf_proxy": vals["rlf_proxy"],
                            "measurement_overhead": vals["measurement_overhead"],
                            "ping_pong_rate": vals["ping_pong_rate"],
                        })
                if shift == "shadow-shift":
                    daci_out = evaluate_daci_robustness_grid(
                        target_data,
                        target_idx,
                        cache,
                        source_cal_scores,
                        args.alpha,
                        daci_gamma_low_grid,
                        daci_gamma_high_grid,
                        daci_ema_beta_grid,
                        args.k_max,
                    )
                    for key, vals in daci_out.items():
                        if key not in daci_robust_seed["shadow-shift"]:
                            daci_robust_seed["shadow-shift"][key] = []
                        daci_robust_seed["shadow-shift"][key].append({
                            "coverage": vals["coverage"],
                            "avg_set_size": vals["avg_set_size"],
                            "rlf_proxy": vals["rlf_proxy"],
                            "measurement_overhead": vals["measurement_overhead"],
                            "ping_pong_rate": vals["ping_pong_rate"],
                        })

            for shift in SHIFT_ORDER:
                for method in METHOD_ORDER:
                    if method not in seed_shift_results[shift]:
                        continue
                    if "seed_metrics" not in aggregated[shift][method]:
                        aggregated[shift][method]["seed_metrics"] = []
                    aggregated[shift][method]["seed_metrics"].append(seed_shift_results[shift][method])

        shift_delta = {}
        for shift in SHIFT_ORDER:
            s_static = aggregated[shift]["static-cp"]["seed_metrics"]
            s_raps = aggregated[shift]["raps-cp"]["seed_metrics"]
            s_aci = aggregated[shift]["aci"]["seed_metrics"]
            s_daci = aggregated[shift]["daci"]["seed_metrics"]
            s_triggered = aggregated[shift]["triggered-aci"]["seed_metrics"]
            s_weighted = aggregated[shift]["weighted-cp"]["seed_metrics"]
            shift_delta[shift] = {
                "aci_minus_static_coverage": paired_delta_with_ci(s_aci, s_static, "coverage"),
                "aci_minus_static_rlf_proxy": paired_delta_with_ci(s_aci, s_static, "rlf_proxy"),
                "raps_minus_static_coverage": paired_delta_with_ci(s_raps, s_static, "coverage"),
                "raps_minus_static_overhead": paired_delta_with_ci(s_raps, s_static, "measurement_overhead"),
                "daci_minus_static_coverage": paired_delta_with_ci(s_daci, s_static, "coverage"),
                "daci_minus_aci_coverage": paired_delta_with_ci(s_daci, s_aci, "coverage"),
                "daci_minus_aci_overhead": paired_delta_with_ci(s_daci, s_aci, "measurement_overhead"),
                "triggered_minus_static_coverage": paired_delta_with_ci(s_triggered, s_static, "coverage"),
                "triggered_minus_aci_coverage": paired_delta_with_ci(s_triggered, s_aci, "coverage"),
                "triggered_minus_aci_overhead": paired_delta_with_ci(s_triggered, s_aci, "measurement_overhead"),
                "weighted_minus_static_coverage": paired_delta_with_ci(s_weighted, s_static, "coverage"),
            }
            s_ensemble = aggregated[shift]["ensemble-cp"].get("seed_metrics", [])
            if s_ensemble:
                shift_delta[shift]["ensemble_minus_static_coverage"] = paired_delta_with_ci(s_ensemble, s_static, "coverage")
                shift_delta[shift]["ensemble_minus_static_overhead"] = paired_delta_with_ci(s_ensemble, s_static, "measurement_overhead")
                shift_delta[shift]["ensemble_minus_aci_coverage"] = paired_delta_with_ci(s_ensemble, s_aci, "coverage")

        for shift in SHIFT_ORDER:
            for method in METHOD_ORDER:
                seed_metrics = aggregated[shift][method].get("seed_metrics", [])
                if not seed_metrics:
                    continue
                aggregated[shift][method] = aggregate_seed_metrics(seed_metrics)

        rolling_out = {}
        for method, series_list in regime_rolling_seed.items():
            roll_series = []
            for s in series_list:
                roll_series.append(rolling_mean(np.array(s, dtype=float), args.rolling_window))
            min_len = min(len(r) for r in roll_series)
            stacked = np.vstack([r[:min_len] for r in roll_series])
            rolling_out[method] = stacked.mean(axis=0).tolist()

        gamma_agg = {}
        for g in gamma_grid:
            key = str(g)
            gamma_agg[key] = aggregate_seed_metrics(gamma_seed[key])

        trigger_agg = {}
        for q in trigger_quantile_grid:
            key = str(q)
            trigger_agg[key] = aggregate_seed_metrics(trigger_seed[key])

        daci_robustness = {}
        for shift in daci_robust_seed.keys():
            daci_robustness[shift] = {}
            for key, metrics in daci_robust_seed[shift].items():
                daci_robustness[shift][key] = aggregate_seed_metrics(metrics)

        conditional_coverage = aggregate_conditional_seed_summaries(conditional_seed)

        render_shift_figures(aggregated, rolling_out, figures_dir, args.alpha)
        render_aci_gamma_figure(gamma_agg, figures_dir, args.alpha)
        render_trigger_quantile_figure(trigger_agg, figures_dir, args.alpha)
        render_hard_shift_pareto(aggregated, figures_dir)
        render_daci_robustness_figure(daci_robustness, figures_dir)
        render_conditional_coverage_figure(
            conditional_coverage["speed_deciles"],
            figures_dir,
            "speed",
            "conditional-coverage-speed-v6",
            "Regime-Switch Coverage by Speed Decile",
        )
        render_conditional_coverage_figure(
            conditional_coverage["confidence_deciles"],
            figures_dir,
            "confidence",
            "conditional-coverage-confidence-v6",
            "Regime-Switch Coverage by Confidence Decile",
        )

        payload["synthetic_shift"] = {
            "aggregated": aggregated,
            "regime_rolling": rolling_out,
            "aci_gamma_ablation": gamma_agg,
            "trigger_quantile_ablation": trigger_agg,
            "daci_robustness": daci_robustness,
            "conditional_coverage": conditional_coverage,
            "ensemble_baseline": {
                "enabled": bool(args.run_ensemble_baseline),
                "ensemble_members": args.ensemble_members,
            },
            "paired_deltas": shift_delta,
        }

    if args.mode in ["irish-shift", "all"]:
        irish_path = figures_dir / "irish-shift-results-v6.json"
        run_irish_shift_script(
            project_dir,
            irish_path,
            args.trigger_quantile,
            args.daci_gamma_low,
            args.daci_gamma_high,
            args.daci_ema_beta,
            args.daci_gamma_low_grid,
            args.daci_gamma_high_grid,
            args.daci_ema_beta_grid,
            args.raps_k_reg,
            args.raps_lam,
            args.irish_split_protocol,
            args.irish_extra_split_protocol,
            args.run_ensemble_baseline,
            args.ensemble_members,
            args.irish_seed,
        )
        payload["irish_shift"] = json.loads(irish_path.read_text())

    if (
        args.run_ensemble_baseline
        and "aggregated" in payload["synthetic_shift"]
        and "results" in payload["irish_shift"]
    ):
        render_ensemble_vs_cp_figure(
            payload["synthetic_shift"]["aggregated"],
            payload["irish_shift"]["results"],
            figures_dir,
        )

    overflow = maybe_run_vast_overflow(
        args.use_vast,
        spent_usd,
        args.cost_soft_stop_usd,
        args.budget_cap_usd,
        project_dir / ".env",
    )

    payload["cost_tracking"] = {
        "spent_usd": spent_usd,
        "budget_cap_usd": args.budget_cap_usd,
        "cost_soft_stop_usd": args.cost_soft_stop_usd,
        "elapsed_seconds": time.time() - start,
        "overflow": overflow,
    }

    out_path = project_dir / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
