import argparse
import json
import os
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
    predict_sets,
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

METHOD_ORDER = ["3db", "top1", "top3", "static-cp", "aci", "daci", "triggered-aci", "weighted-cp"]


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
        vals = [m[k] for m in seed_metrics]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
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
    trigger_quantile,
    device,
    k_max,
):
    probs = probs_with_stats(model, target_data, target_idx, stats, device)
    labels = target_data["optimal_cell"][target_idx]

    top1_pred = np.argmax(probs, axis=1)
    top1_sets = [np.array([int(v)]) for v in top1_pred]
    top3 = topk_sets(probs, 3)

    threshold = calibrate_threshold(source_cal_probs, source_cal_labels, alpha=alpha)
    sets_static = predict_sets(probs, threshold)

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

    baseline = evaluate_3db_baseline(target_data["rsrp"][target_idx], target_data["serving_cell"][target_idx], labels)

    out = {}

    for method, sets, policy in [
        ("top1", top1_sets, "ml_top1"),
        ("top3", top3, "cp_adaptive"),
        ("static-cp", sets_static, "cp_adaptive"),
        ("aci", sets_aci, "cp_adaptive"),
        ("daci", sets_daci, "cp_adaptive"),
        ("triggered-aci", sets_triggered, "cp_adaptive"),
        ("weighted-cp", sets_weighted, "cp_adaptive"),
    ]:
        ev = evaluate_cp(sets, labels)
        ho = simulate_handover_protocol(sets, labels, int(target_data["n_cells"]), k_max=k_max)
        pp = ping_pong_rate_for_policy(target_data, target_idx, policy, ml_predictions=top1_pred, pred_sets=sets, k_max=k_max)
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
        "static-cp": [int(labels[i] in sets_static[i]) for i in order],
        "aci": [int(covered_aci[i]) for i in order],
        "daci": [int(covered_daci[i]) for i in order],
        "triggered-aci": [int(covered_triggered[i]) for i in order],
        "weighted-cp": [int(labels[i] in sets_weighted[i]) for i in order],
    }

    cache = {
        "probs": probs,
        "labels": labels,
        "order": order,
        "top1_pred": top1_pred,
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
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for i, method in enumerate(["top1", "static-cp", "aci", "weighted-cp"]):
        means = [aggregated[s][method]["coverage_mean"] for s in shift_labels]
        errs = [aggregated[s][method]["coverage_std"] for s in shift_labels]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=errs, capsize=2, label=method)
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

    methods = ["static-cp", "aci", "daci", "weighted-cp"]
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


def run_irish_shift_script(
    project_dir: Path,
    output_json: Path,
    trigger_quantile: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
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
    ]
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
    parser.add_argument("--trigger-quantile", type=float, default=0.7)
    parser.add_argument("--aci-gamma-grid", type=str, default="0.002,0.005,0.01,0.02,0.05")
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
    if args.trigger_quantile < 0.0 or args.trigger_quantile > 1.0:
        raise ValueError("--trigger-quantile must be in [0,1]")
    if args.daci_gamma_low <= 0.0 or args.daci_gamma_high <= 0.0:
        raise ValueError("--daci-gamma-low and --daci-gamma-high must be > 0")
    if args.daci_gamma_low > args.daci_gamma_high:
        raise ValueError("--daci-gamma-low must be <= --daci-gamma-high")
    if args.daci_ema_beta < 0.0 or args.daci_ema_beta >= 1.0:
        raise ValueError("--daci-ema-beta must be in [0,1)")

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
            "trigger_quantile": args.trigger_quantile,
            "k_max": args.k_max,
            "device": device,
            "budget_cap_usd": args.budget_cap_usd,
            "cost_soft_stop_usd": args.cost_soft_stop_usd,
            "strategy": "local-first",
        },
        "synthetic_shift": {},
        "irish_shift": {},
        "cost_tracking": {},
    }

    if args.mode in ["synthetic-shift", "all"]:
        aggregated = {s: {m: {} for m in METHOD_ORDER} for s in SHIFT_ORDER}
        regime_rolling_seed = {"static-cp": [], "aci": [], "daci": [], "weighted-cp": []}
        gamma_seed = {str(g): [] for g in gamma_grid}

        for seed in seeds:
            source_data = generate_shift_data(seed, args.n_traj, SHIFT_CONFIGS["iid"])
            train_idx, cal_idx, _ = split_by_trajectory(source_data["trajectory_id"], args.n_traj)
            source_idx = {"train": train_idx, "cal": cal_idx}

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

            seed_shift_results = {}

            for shift in SHIFT_ORDER:
                if shift == "regime-switch":
                    target_data = make_regime_switch_data(seed, args.n_traj)
                    target_idx = np.arange(len(target_data["optimal_cell"]))
                else:
                    target_data = generate_shift_data(seed + 1000 + SHIFT_ORDER.index(shift), args.n_traj, SHIFT_CONFIGS[shift])
                    _, _, test_idx = split_by_trajectory(target_data["trajectory_id"], args.n_traj)
                    target_idx = test_idx

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
                    args.trigger_quantile,
                    device,
                    args.k_max,
                )
                seed_shift_results[shift] = methods_out

                if shift == "regime-switch":
                    for method in regime_rolling_seed:
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
            s_aci = aggregated[shift]["aci"]["seed_metrics"]
            s_daci = aggregated[shift]["daci"]["seed_metrics"]
            s_triggered = aggregated[shift]["triggered-aci"]["seed_metrics"]
            s_weighted = aggregated[shift]["weighted-cp"]["seed_metrics"]
            shift_delta[shift] = {
                "aci_minus_static_coverage": paired_delta_with_ci(s_aci, s_static, "coverage"),
                "aci_minus_static_rlf_proxy": paired_delta_with_ci(s_aci, s_static, "rlf_proxy"),
                "daci_minus_static_coverage": paired_delta_with_ci(s_daci, s_static, "coverage"),
                "daci_minus_aci_coverage": paired_delta_with_ci(s_daci, s_aci, "coverage"),
                "daci_minus_aci_overhead": paired_delta_with_ci(s_daci, s_aci, "measurement_overhead"),
                "triggered_minus_static_coverage": paired_delta_with_ci(s_triggered, s_static, "coverage"),
                "triggered_minus_aci_coverage": paired_delta_with_ci(s_triggered, s_aci, "coverage"),
                "triggered_minus_aci_overhead": paired_delta_with_ci(s_triggered, s_aci, "measurement_overhead"),
                "weighted_minus_static_coverage": paired_delta_with_ci(s_weighted, s_static, "coverage"),
            }

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

        render_shift_figures(aggregated, rolling_out, figures_dir, args.alpha)
        render_aci_gamma_figure(gamma_agg, figures_dir, args.alpha)

        payload["synthetic_shift"] = {
            "aggregated": aggregated,
            "regime_rolling": rolling_out,
            "aci_gamma_ablation": gamma_agg,
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
        )
        payload["irish_shift"] = json.loads(irish_path.read_text())

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
