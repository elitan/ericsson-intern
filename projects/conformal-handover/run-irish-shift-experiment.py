import argparse
import datetime
import json
import random
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_irish_experiment import train_model, get_probs
from src.handover.conformal import (
    AdaptiveConformalInference,
    calibrate_threshold,
    calibrate_weighted_threshold,
    calibrate_raps,
    estimate_density_ratio_weights,
    evaluate_cp,
    predict_sets,
    predict_sets_raps,
)
from src.handover.irish_data import load_driving_traces, preprocess_for_handover


def ensure_data(project_dir: Path):
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

    raise RuntimeError("Irish dataset not found after clone")


def rolling_mean(values: np.ndarray, window: int):
    if len(values) < window:
        return np.array([values.mean()])
    c = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


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


def build_trace_split(trace_ids: np.ndarray, speed_col: np.ndarray, seed: int, protocol: str):
    unique_traces = np.unique(trace_ids)
    if protocol == "speed-split":
        trace_speed = []
        for t in unique_traces:
            mask = trace_ids == t
            trace_speed.append((t, float(speed_col[mask].mean())))
        trace_speed.sort(key=lambda x: x[1])
        half = len(trace_speed) // 2
        source_traces = [t for t, _ in trace_speed[:half]]
        target_traces = [t for t, _ in trace_speed[half:]]
        source_traces = np.array(source_traces)
        rng = np.random.default_rng(seed)
        rng.shuffle(source_traces)
        split = int(0.7 * len(source_traces))
        train_traces = set(source_traces[:split].tolist())
        cal_traces = set(source_traces[split:].tolist())
        target_traces = set(target_traces)
    elif protocol == "trace-holdout":
        traces = np.array(unique_traces)
        rng = np.random.default_rng(seed)
        rng.shuffle(traces)
        n = len(traces)
        n_train = int(0.6 * n)
        n_cal = int(0.2 * n)
        train_traces = set(traces[:n_train].tolist())
        cal_traces = set(traces[n_train:n_train + n_cal].tolist())
        target_traces = set(traces[n_train + n_cal:].tolist())
    else:
        raise ValueError(f"unknown split protocol: {protocol}")

    train_idx = np.array([i for i, t in enumerate(trace_ids) if t in train_traces])
    cal_idx = np.array([i for i, t in enumerate(trace_ids) if t in cal_traces])
    target_idx = np.array([i for i, t in enumerate(trace_ids) if t in target_traces])
    return {
        "protocol": protocol,
        "train_traces": train_traces,
        "cal_traces": cal_traces,
        "target_traces": target_traces,
        "train_idx": train_idx,
        "cal_idx": cal_idx,
        "target_idx": target_idx,
    }


def topk_sets(probs: np.ndarray, k: int):
    idx = np.argsort(probs, axis=1)[:, -k:]
    out = []
    for i in range(len(probs)):
        out.append(np.array(sorted(idx[i].tolist())))
    return out


def set_cover_array(labels: np.ndarray, sets: list):
    return np.array([int(labels[i] in sets[i]) for i in range(len(labels))], dtype=float)


def set_size_array(sets: list):
    return np.array([len(sets[i]) for i in range(len(sets))], dtype=float)


def ordered_cover_list(labels: np.ndarray, sets: list, order: np.ndarray):
    return [int(labels[i] in sets[i]) for i in order]


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


def summarize_sets(name: str, sets: list, labels: np.ndarray):
    ev = evaluate_cp(sets, labels)
    return {
        "method": name,
        "coverage": float(ev["coverage"]),
        "avg_set_size": float(ev["avg_set_size"]),
        "set_size_std": float(ev["set_size_std"]),
    }


def trace_bootstrap_ci(trace_ids: np.ndarray, cover: np.ndarray, sizes: np.ndarray, n_boot: int = 2000, seed: int = 123):
    uniq = np.unique(trace_ids)
    if len(uniq) == 0:
        raise ValueError("empty trace ids")
    idx_by_trace = {}
    for t in uniq:
        idx_by_trace[int(t)] = np.where(trace_ids == t)[0]
    rng = np.random.default_rng(seed)
    cover_boot = []
    size_boot = []
    n = len(uniq)
    uniq_list = [int(x) for x in uniq.tolist()]
    for _ in range(n_boot):
        sampled = rng.choice(uniq_list, size=n, replace=True)
        idx = np.concatenate([idx_by_trace[int(t)] for t in sampled])
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


def trace_bootstrap_delta_ci(trace_ids: np.ndarray, cover_a: np.ndarray, cover_b: np.ndarray, sizes_a: np.ndarray, sizes_b: np.ndarray, n_boot: int = 2000, seed: int = 123):
    uniq = np.unique(trace_ids)
    if len(uniq) == 0:
        raise ValueError("empty trace ids")
    idx_by_trace = {}
    for t in uniq:
        idx_by_trace[int(t)] = np.where(trace_ids == t)[0]
    rng = np.random.default_rng(seed)
    cov_boot = []
    size_boot = []
    n = len(uniq)
    uniq_list = [int(x) for x in uniq.tolist()]
    for _ in range(n_boot):
        sampled = rng.choice(uniq_list, size=n, replace=True)
        idx = np.concatenate([idx_by_trace[int(t)] for t in sampled])
        cov_boot.append(float(np.mean(cover_a[idx]) - np.mean(cover_b[idx])))
        size_boot.append(float(np.mean(sizes_a[idx]) - np.mean(sizes_b[idx])))
    c_lo, c_hi = np.percentile(np.array(cov_boot), [2.5, 97.5])
    s_lo, s_hi = np.percentile(np.array(size_boot), [2.5, 97.5])
    return {
        "coverage_delta_mean": float(np.mean(cover_a) - np.mean(cover_b)),
        "coverage_ci95_low": float(c_lo),
        "coverage_ci95_high": float(c_hi),
        "avg_set_size_delta_mean": float(np.mean(sizes_a) - np.mean(sizes_b)),
        "avg_set_size_ci95_low": float(s_lo),
        "avg_set_size_ci95_high": float(s_hi),
    }


def speed_bin_summary(speeds: np.ndarray, method_cover: dict, method_size: dict):
    q = np.quantile(speeds, [0.33, 0.66])
    bins = {
        "low": np.where(speeds < q[0])[0],
        "mid": np.where((speeds >= q[0]) & (speeds < q[1]))[0],
        "high": np.where(speeds >= q[1])[0],
    }
    out = {
        "cuts": {"q33": float(q[0]), "q66": float(q[1])},
        "bins": {},
    }
    for b, idx in bins.items():
        out["bins"][b] = {}
        for m in method_cover.keys():
            out["bins"][b][m] = {
                "n": int(len(idx)),
                "coverage": float(np.mean(method_cover[m][idx])) if len(idx) else float("nan"),
                "avg_set_size": float(np.mean(method_size[m][idx])) if len(idx) else float("nan"),
            }
    return out


def evaluate_split_summary(
    features: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    speed_col: np.ndarray,
    n_cells: int,
    split: dict,
    alpha: float,
    gamma: float,
    daci_gamma_low: float,
    daci_gamma_high: float,
    daci_ema_beta: float,
    trigger_quantile: float,
    run_ensemble_baseline: bool,
    ensemble_members: int,
    seed: int,
    raps_k_reg: int,
    raps_lam: float,
):
    train_idx = split["train_idx"]
    cal_idx = split["cal_idx"]
    target_idx = split["target_idx"]

    model = train_model(features, labels, train_idx, cal_idx, n_cells, n_epochs=30)
    cal_probs = get_probs(model, features, cal_idx)
    target_probs = get_probs(model, features, target_idx)
    ensemble_cal_probs = None
    ensemble_target_probs = None
    if run_ensemble_baseline:
        cal_stack = [cal_probs]
        target_stack = [target_probs]
        for member in range(1, ensemble_members):
            set_global_seed(seed * 1000 + member)
            model_member = train_model(features, labels, train_idx, cal_idx, n_cells, n_epochs=30)
            cal_stack.append(get_probs(model_member, features, cal_idx))
            target_stack.append(get_probs(model_member, features, target_idx))
        ensemble_cal_probs = np.mean(np.stack(cal_stack, axis=0), axis=0)
        ensemble_target_probs = np.mean(np.stack(target_stack, axis=0), axis=0)
    cal_labels = labels[cal_idx]
    target_labels = labels[target_idx]

    top1 = [np.array([int(v)]) for v in np.argmax(target_probs, axis=1)]
    top3 = topk_sets(target_probs, 3)
    threshold_static = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    sets_static = predict_sets(target_probs, threshold_static)
    threshold_raps = calibrate_raps(cal_probs, cal_labels, alpha=alpha, k_reg=raps_k_reg, lam=raps_lam, rand=False)
    sets_raps = predict_sets_raps(target_probs, threshold_raps, k_reg=raps_k_reg, lam=raps_lam, rand=False)

    weights = estimate_density_ratio_weights(features[train_idx], features[target_idx], features[cal_idx])
    threshold_weighted = calibrate_weighted_threshold(cal_probs, cal_labels, weights, alpha=alpha)
    sets_weighted = predict_sets(target_probs, threshold_weighted)

    cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    order = np.lexsort((target_idx, trace_ids[target_idx]))
    sets_aci, _ = build_aci_sets(target_probs, target_labels, order, cal_scores, alpha, gamma)
    sets_daci, _ = build_daci_sets(
        target_probs,
        target_labels,
        order,
        cal_scores,
        alpha,
        daci_gamma_low,
        daci_gamma_high,
        daci_ema_beta,
    )
    source_conf = np.max(cal_probs, axis=1)
    trigger_tau = float(np.quantile(source_conf, trigger_quantile))
    target_conf = np.max(target_probs, axis=1)
    sets_triggered = []
    for i in range(len(target_labels)):
        use_aci = target_conf[i] < trigger_tau
        sets_triggered.append(sets_aci[i] if use_aci else sets_static[i])
    sets_ensemble = None
    if run_ensemble_baseline:
        threshold_ensemble = calibrate_threshold(ensemble_cal_probs, cal_labels, alpha=alpha)
        sets_ensemble = predict_sets(ensemble_target_probs, threshold_ensemble)

    method_sets = {
        "top1": top1,
        "top3": top3,
        "static-cp": sets_static,
        "raps-cp": sets_raps,
        "aci": sets_aci,
        "daci": sets_daci,
        "triggered-aci": sets_triggered,
        "weighted-cp": sets_weighted,
    }
    if sets_ensemble is not None:
        method_sets["ensemble-cp"] = sets_ensemble

    target_trace_ids = trace_ids[target_idx]
    method_cover = {}
    method_size = {}
    results = {}
    bootstrap = {}
    for name, sets in method_sets.items():
        cover = set_cover_array(target_labels, sets)
        size = set_size_array(sets)
        method_cover[name] = cover
        method_size[name] = size
        results[name] = summarize_sets(name, sets, target_labels)
        bootstrap[name] = trace_bootstrap_ci(target_trace_ids, cover, size)

    paired = {
        "aci_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["aci"],
            method_cover["static-cp"],
            method_size["aci"],
            method_size["static-cp"],
        ),
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
    }

    return {
        "protocol": split["protocol"],
        "split": {
            "protocol": split["protocol"],
            "train": int(len(train_idx)),
            "cal": int(len(cal_idx)),
            "target": int(len(target_idx)),
        },
        "trace_counts": {
            "source": int(len(split["train_traces"]) + len(split["cal_traces"])),
            "target": int(len(split["target_traces"])),
        },
        "results": results,
        "bootstrap_ci": bootstrap,
        "paired_deltas": paired,
        "speed_summary": {
            "target_speed_mean": float(np.mean(speed_col[target_idx])) if len(target_idx) else float("nan"),
            "target_speed_std": float(np.std(speed_col[target_idx])) if len(target_idx) else float("nan"),
        },
    }


def evaluate_daci_robustness_grid(
    probs: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float,
    gamma_low_grid: list,
    gamma_high_grid: list,
    ema_beta_grid: list,
):
    out = {}
    for gamma_low in gamma_low_grid:
        for gamma_high in gamma_high_grid:
            if gamma_low > gamma_high:
                continue
            for ema_beta in ema_beta_grid:
                sets, _ = build_daci_sets(
                    probs,
                    labels,
                    order,
                    cal_scores,
                    alpha,
                    gamma_low,
                    gamma_high,
                    ema_beta,
                )
                ev = evaluate_cp(sets, labels)
                key = f"gamma_low={gamma_low}|gamma_high={gamma_high}|ema_beta={ema_beta}"
                out[key] = {
                    "coverage": float(ev["coverage"]),
                    "avg_set_size": float(ev["avg_set_size"]),
                    "set_size_std": float(ev["set_size_std"]),
                }
    return out


def render_conditional_figure(conditional: dict, figures_dir: Path, domain: str, out_name: str, title: str):
    methods = list(conditional["deciles"][0]["methods"].keys())
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for method in methods:
        y = [conditional["deciles"][i]["methods"][method] for i in range(10)]
        ax.plot(x, y, marker="o", linewidth=1.6, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(f"{domain} decile")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.55, 0.97)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{out_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, default="figures/irish-shift-results-v6.json")
    parser.add_argument("--max-files", type=int, default=50)
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
    parser.add_argument("--split-protocol", choices=["speed-split", "trace-holdout"], default="speed-split")
    parser.add_argument("--extra-split-protocol", choices=["none", "trace-holdout"], default="trace-holdout")
    parser.add_argument("--run-ensemble-baseline", action="store_true")
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trigger_quantile < 0.0 or args.trigger_quantile > 1.0:
        raise ValueError("--trigger-quantile must be in [0,1]")
    if args.daci_gamma_low <= 0.0 or args.daci_gamma_high <= 0.0:
        raise ValueError("--daci-gamma-low and --daci-gamma-high must be > 0")
    if args.daci_gamma_low > args.daci_gamma_high:
        raise ValueError("--daci-gamma-low must be <= --daci-gamma-high")
    if args.daci_ema_beta < 0.0 or args.daci_ema_beta >= 1.0:
        raise ValueError("--daci-ema-beta must be in [0,1)")
    if args.raps_k_reg < 1:
        raise ValueError("--raps-k-reg must be >= 1")
    if args.raps_lam < 0.0:
        raise ValueError("--raps-lam must be >= 0")
    if args.ensemble_members < 1:
        raise ValueError("--ensemble-members must be >= 1")
    daci_gamma_low_grid = parse_float_list(args.daci_gamma_low_grid)
    daci_gamma_high_grid = parse_float_list(args.daci_gamma_high_grid)
    daci_ema_beta_grid = parse_float_list(args.daci_ema_beta_grid)
    if not daci_gamma_low_grid or not daci_gamma_high_grid or not daci_ema_beta_grid:
        raise ValueError("daci grids must be non-empty")
    for val in daci_gamma_low_grid + daci_gamma_high_grid:
        if val <= 0.0:
            raise ValueError("daci gamma grid values must be > 0")
    for val in daci_ema_beta_grid:
        if val < 0.0 or val >= 1.0:
            raise ValueError("daci ema beta grid values must be in [0,1)")

    set_global_seed(args.seed)

    project_dir = Path(__file__).resolve().parent
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    data_dir = ensure_data(project_dir)
    df = load_driving_traces(data_dir, max_files=args.max_files)
    data = preprocess_for_handover(df)

    features = data["features"]
    labels = data["next_cell"]
    trace_ids = data["trace_id"]

    speed_col = features[:, 2]
    split_info = build_trace_split(trace_ids, speed_col, args.seed, args.split_protocol)
    train_traces = split_info["train_traces"]
    cal_traces = split_info["cal_traces"]
    target_traces = split_info["target_traces"]
    train_idx = split_info["train_idx"]
    cal_idx = split_info["cal_idx"]
    target_idx = split_info["target_idx"]

    model = train_model(features, labels, train_idx, cal_idx, data["n_cells"], n_epochs=30)

    cal_probs = get_probs(model, features, cal_idx)
    target_probs = get_probs(model, features, target_idx)
    ensemble_cal_probs = None
    ensemble_target_probs = None
    if args.run_ensemble_baseline:
        cal_stack = [cal_probs]
        target_stack = [target_probs]
        for member in range(1, args.ensemble_members):
            set_global_seed(args.seed * 1000 + member)
            model_member = train_model(features, labels, train_idx, cal_idx, data["n_cells"], n_epochs=30)
            cal_stack.append(get_probs(model_member, features, cal_idx))
            target_stack.append(get_probs(model_member, features, target_idx))
        ensemble_cal_probs = np.mean(np.stack(cal_stack, axis=0), axis=0)
        ensemble_target_probs = np.mean(np.stack(target_stack, axis=0), axis=0)
    cal_labels = labels[cal_idx]
    target_labels = labels[target_idx]

    top1 = [np.array([int(v)]) for v in np.argmax(target_probs, axis=1)]
    top3 = topk_sets(target_probs, 3)

    threshold_static = calibrate_threshold(cal_probs, cal_labels, alpha=args.alpha)
    sets_static = predict_sets(target_probs, threshold_static)
    threshold_raps = calibrate_raps(
        cal_probs,
        cal_labels,
        alpha=args.alpha,
        k_reg=args.raps_k_reg,
        lam=args.raps_lam,
        rand=False,
    )
    sets_raps = predict_sets_raps(
        target_probs,
        threshold_raps,
        k_reg=args.raps_k_reg,
        lam=args.raps_lam,
        rand=False,
    )

    weights = estimate_density_ratio_weights(features[train_idx], features[target_idx], features[cal_idx])
    threshold_weighted = calibrate_weighted_threshold(cal_probs, cal_labels, weights, alpha=args.alpha)
    sets_weighted = predict_sets(target_probs, threshold_weighted)

    cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    order = np.lexsort((target_idx, trace_ids[target_idx]))
    sets_aci, covered_aci = build_aci_sets(target_probs, target_labels, order, cal_scores, args.alpha, args.gamma)
    sets_daci, covered_daci = build_daci_sets(
        target_probs,
        target_labels,
        order,
        cal_scores,
        args.alpha,
        args.daci_gamma_low,
        args.daci_gamma_high,
        args.daci_ema_beta,
    )
    source_conf = np.max(cal_probs, axis=1)
    trigger_tau = float(np.quantile(source_conf, args.trigger_quantile))
    target_conf = np.max(target_probs, axis=1)
    sets_triggered = []
    covered_triggered = np.zeros(len(target_labels), dtype=bool)
    for i in range(len(target_labels)):
        use_aci = target_conf[i] < trigger_tau
        s = sets_aci[i] if use_aci else sets_static[i]
        sets_triggered.append(s)
        covered_triggered[i] = bool(target_labels[i] in s)
    sets_ensemble = None
    if args.run_ensemble_baseline:
        threshold_ensemble = calibrate_threshold(ensemble_cal_probs, cal_labels, alpha=args.alpha)
        sets_ensemble = predict_sets(ensemble_target_probs, threshold_ensemble)

    rolling = {
        "static-cp": rolling_mean(np.array(ordered_cover_list(target_labels, sets_static, order), dtype=float), args.rolling_window).tolist(),
        "raps-cp": rolling_mean(np.array(ordered_cover_list(target_labels, sets_raps, order), dtype=float), args.rolling_window).tolist(),
        "aci": rolling_mean(covered_aci[order].astype(float), args.rolling_window).tolist(),
        "daci": rolling_mean(covered_daci[order].astype(float), args.rolling_window).tolist(),
        "triggered-aci": rolling_mean(covered_triggered[order].astype(float), args.rolling_window).tolist(),
        "weighted-cp": rolling_mean(np.array(ordered_cover_list(target_labels, sets_weighted, order), dtype=float), args.rolling_window).tolist(),
    }
    if sets_ensemble is not None:
        rolling["ensemble-cp"] = rolling_mean(np.array(ordered_cover_list(target_labels, sets_ensemble, order), dtype=float), args.rolling_window).tolist()
    target_trace_ids = trace_ids[target_idx]
    method_sets = {
        "top1": top1,
        "top3": top3,
        "static-cp": sets_static,
        "raps-cp": sets_raps,
        "aci": sets_aci,
        "daci": sets_daci,
        "triggered-aci": sets_triggered,
        "weighted-cp": sets_weighted,
    }
    if sets_ensemble is not None:
        method_sets["ensemble-cp"] = sets_ensemble
    method_cover = {}
    method_size = {}
    for name, sets in method_sets.items():
        method_cover[name] = set_cover_array(target_labels, sets)
        method_size[name] = set_size_array(sets)
    bootstrap = {}
    for name in method_sets:
        bootstrap[name] = trace_bootstrap_ci(target_trace_ids, method_cover[name], method_size[name])
    speed_bins = speed_bin_summary(speed_col[target_idx], method_cover, method_size)
    conditional_coverage = {
        "speed_deciles": conditional_coverage_summary(speed_col[target_idx], method_cover),
        "confidence_deciles": conditional_coverage_summary(target_conf, method_cover),
    }
    daci_robustness = evaluate_daci_robustness_grid(
        target_probs,
        target_labels,
        order,
        cal_scores,
        args.alpha,
        daci_gamma_low_grid,
        daci_gamma_high_grid,
        daci_ema_beta_grid,
    )
    paired_deltas = {
        "aci_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["aci"],
            method_cover["static-cp"],
            method_size["aci"],
            method_size["static-cp"],
        ),
        "daci_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["daci"],
            method_cover["static-cp"],
            method_size["daci"],
            method_size["static-cp"],
        ),
        "daci_minus_aci": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["daci"],
            method_cover["aci"],
            method_size["daci"],
            method_size["aci"],
        ),
        "raps_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["raps-cp"],
            method_cover["static-cp"],
            method_size["raps-cp"],
            method_size["static-cp"],
        ),
        "triggered_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["triggered-aci"],
            method_cover["static-cp"],
            method_size["triggered-aci"],
            method_size["static-cp"],
        ),
        "triggered_minus_aci": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["triggered-aci"],
            method_cover["aci"],
            method_size["triggered-aci"],
            method_size["aci"],
        ),
    }
    if "ensemble-cp" in method_cover:
        paired_deltas["ensemble_minus_static"] = trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["ensemble-cp"],
            method_cover["static-cp"],
            method_size["ensemble-cp"],
            method_size["static-cp"],
        )
        paired_deltas["ensemble_minus_aci"] = trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["ensemble-cp"],
            method_cover["aci"],
            method_size["ensemble-cp"],
            method_size["aci"],
        )

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
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    plot_methods = ["static-cp", "raps-cp", "aci", "daci", "triggered-aci", "weighted-cp"]
    if "ensemble-cp" in rolling:
        plot_methods.append("ensemble-cp")
    for method in plot_methods:
        ax.plot(rolling[method], label=method, linewidth=1.8)
    ax.axhline(1 - args.alpha, color="red", linestyle="--", linewidth=1.2)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Rolling coverage")
    ax.set_title("Irish Drift Rolling Coverage (window=200)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "irish-shift-rolling-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "irish-shift-rolling-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    speed_bin_labels = ["low", "mid", "high"]
    x = np.arange(len(speed_bin_labels))
    bin_methods = ["static-cp", "raps-cp", "aci", "daci", "triggered-aci", "weighted-cp"]
    if "ensemble-cp" in speed_bins["bins"]["low"]:
        bin_methods.append("ensemble-cp")
    width = 0.8 / max(len(bin_methods), 1)
    for i, method in enumerate(bin_methods):
        y = [speed_bins["bins"][b][method]["coverage"] for b in speed_bin_labels]
        ax.bar(x + (i - (len(bin_methods) - 1) / 2.0) * width, y, width, label=method)
    ax.axhline(1 - args.alpha, color="red", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Irish Target Coverage by Speed Bin")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "irish-speed-bins-v6.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "irish-speed-bins-v6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    results_out = {
        "top1": summarize_sets("top1", top1, target_labels),
        "top3": summarize_sets("top3", top3, target_labels),
        "static-cp": summarize_sets("static-cp", sets_static, target_labels),
        "raps-cp": summarize_sets("raps-cp", sets_raps, target_labels),
        "aci": summarize_sets("aci", sets_aci, target_labels),
        "daci": summarize_sets("daci", sets_daci, target_labels),
        "triggered-aci": summarize_sets("triggered-aci", sets_triggered, target_labels),
        "weighted-cp": summarize_sets("weighted-cp", sets_weighted, target_labels),
    }
    if sets_ensemble is not None:
        results_out["ensemble-cp"] = summarize_sets("ensemble-cp", sets_ensemble, target_labels)

    additional_splits = {}
    if args.extra_split_protocol != "none" and args.extra_split_protocol != args.split_protocol:
        extra_split = build_trace_split(trace_ids, speed_col, args.seed + 17, args.extra_split_protocol)
        additional_splits[args.extra_split_protocol] = evaluate_split_summary(
            features,
            labels,
            trace_ids,
            speed_col,
            int(data["n_cells"]),
            extra_split,
            args.alpha,
            args.gamma,
            args.daci_gamma_low,
            args.daci_gamma_high,
            args.daci_ema_beta,
            args.trigger_quantile,
            bool(args.run_ensemble_baseline),
            args.ensemble_members,
            args.seed + 17,
            args.raps_k_reg,
            args.raps_lam,
        )

    result = {
        "metadata": {
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
            "split_protocol": args.split_protocol,
            "extra_split_protocol": args.extra_split_protocol,
            "run_ensemble_baseline": bool(args.run_ensemble_baseline),
            "ensemble_members": args.ensemble_members,
            "seed": args.seed,
            "max_files": args.max_files,
            "dataset_dir": str(data_dir),
            "n_cells": int(data["n_cells"]),
            "n_samples": int(len(features)),
            "n_source_traces": int(len(train_traces) + len(cal_traces)),
            "n_target_traces": int(len(target_traces)),
            "created_at_iso_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "git_commit": get_git_commit(project_dir),
            "argv": sys.argv,
        },
        "split": {
            "train": int(len(train_idx)),
            "cal": int(len(cal_idx)),
            "target": int(len(target_idx)),
        },
        "results": results_out,
        "bootstrap_ci": bootstrap,
        "speed_bins": speed_bins,
        "paired_deltas": paired_deltas,
        "rolling": rolling,
        "daci_robustness": daci_robustness,
        "conditional_coverage": conditional_coverage,
        "additional_splits": additional_splits,
        "ensemble_baseline": {
            "enabled": bool(args.run_ensemble_baseline),
            "ensemble_members": args.ensemble_members,
        },
    }

    out_path = project_dir / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
