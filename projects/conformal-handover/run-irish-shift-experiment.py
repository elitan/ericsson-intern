import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_irish_experiment import train_model, get_probs
from src.handover.conformal import (
    AdaptiveConformalInference,
    calibrate_threshold,
    calibrate_weighted_threshold,
    estimate_density_ratio_weights,
    evaluate_cp,
    predict_sets,
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


def topk_sets(probs: np.ndarray, k: int):
    idx = np.argsort(probs, axis=1)[:, -k:]
    out = []
    for i in range(len(probs)):
        out.append(np.array(sorted(idx[i].tolist())))
    return out


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, default="figures/irish-shift-results-v6.json")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--trigger-quantile", type=float, default=0.7)
    parser.add_argument("--rolling-window", type=int, default=200)
    args = parser.parse_args()
    if args.trigger_quantile < 0.0 or args.trigger_quantile > 1.0:
        raise ValueError("--trigger-quantile must be in [0,1]")

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
    unique_traces = np.unique(trace_ids)
    trace_speed = []
    for t in unique_traces:
        mask = trace_ids == t
        trace_speed.append((t, float(speed_col[mask].mean())))
    trace_speed.sort(key=lambda x: x[1])

    half = len(trace_speed) // 2
    source_traces = [t for t, _ in trace_speed[:half]]
    target_traces = [t for t, _ in trace_speed[half:]]

    source_traces = np.array(source_traces)
    rng = np.random.default_rng(42)
    rng.shuffle(source_traces)

    split = int(0.7 * len(source_traces))
    train_traces = set(source_traces[:split].tolist())
    cal_traces = set(source_traces[split:].tolist())
    target_traces = set(target_traces)

    train_idx = np.array([i for i, t in enumerate(trace_ids) if t in train_traces])
    cal_idx = np.array([i for i, t in enumerate(trace_ids) if t in cal_traces])
    target_idx = np.array([i for i, t in enumerate(trace_ids) if t in target_traces])

    model = train_model(features, labels, train_idx, cal_idx, data["n_cells"], n_epochs=30)

    cal_probs = get_probs(model, features, cal_idx)
    target_probs = get_probs(model, features, target_idx)
    cal_labels = labels[cal_idx]
    target_labels = labels[target_idx]

    top1 = [np.array([int(v)]) for v in np.argmax(target_probs, axis=1)]
    top3 = topk_sets(target_probs, 3)

    threshold_static = calibrate_threshold(cal_probs, cal_labels, alpha=args.alpha)
    sets_static = predict_sets(target_probs, threshold_static)

    weights = estimate_density_ratio_weights(features[train_idx], features[target_idx], features[cal_idx])
    threshold_weighted = calibrate_weighted_threshold(cal_probs, cal_labels, weights, alpha=args.alpha)
    sets_weighted = predict_sets(target_probs, threshold_weighted)

    cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    order = np.lexsort((target_idx, trace_ids[target_idx]))
    sets_aci, covered_aci = build_aci_sets(target_probs, target_labels, order, cal_scores, args.alpha, args.gamma)
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

    rolling = {
        "static-cp": rolling_mean(np.array([int(target_labels[i] in sets_static[i]) for i in order]), args.rolling_window).tolist(),
        "aci": rolling_mean(covered_aci[order].astype(float), args.rolling_window).tolist(),
        "triggered-aci": rolling_mean(covered_triggered[order].astype(float), args.rolling_window).tolist(),
        "weighted-cp": rolling_mean(np.array([int(target_labels[i] in sets_weighted[i]) for i in order]), args.rolling_window).tolist(),
    }
    target_trace_ids = trace_ids[target_idx]
    method_sets = {
        "top1": top1,
        "top3": top3,
        "static-cp": sets_static,
        "aci": sets_aci,
        "triggered-aci": sets_triggered,
        "weighted-cp": sets_weighted,
    }
    method_cover = {}
    method_size = {}
    for name, sets in method_sets.items():
        method_cover[name] = np.array([int(target_labels[i] in sets[i]) for i in range(len(target_labels))], dtype=float)
        method_size[name] = np.array([len(sets[i]) for i in range(len(target_labels))], dtype=float)
    bootstrap = {}
    for name in method_sets:
        bootstrap[name] = trace_bootstrap_ci(target_trace_ids, method_cover[name], method_size[name])
    paired_deltas = {
        "aci_minus_static": trace_bootstrap_delta_ci(
            target_trace_ids,
            method_cover["aci"],
            method_cover["static-cp"],
            method_size["aci"],
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
    for method in ["static-cp", "aci", "triggered-aci", "weighted-cp"]:
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

    result = {
        "metadata": {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "trigger_quantile": args.trigger_quantile,
            "max_files": args.max_files,
            "dataset_dir": str(data_dir),
            "n_cells": int(data["n_cells"]),
            "n_samples": int(len(features)),
            "n_source_traces": int(len(train_traces) + len(cal_traces)),
            "n_target_traces": int(len(target_traces)),
        },
        "split": {
            "train": int(len(train_idx)),
            "cal": int(len(cal_idx)),
            "target": int(len(target_idx)),
        },
        "results": {
            "top1": summarize_sets("top1", top1, target_labels),
            "top3": summarize_sets("top3", top3, target_labels),
            "static-cp": summarize_sets("static-cp", sets_static, target_labels),
            "aci": summarize_sets("aci", sets_aci, target_labels),
            "triggered-aci": summarize_sets("triggered-aci", sets_triggered, target_labels),
            "weighted-cp": summarize_sets("weighted-cp", sets_weighted, target_labels),
        },
        "bootstrap_ci": bootstrap,
        "paired_deltas": paired_deltas,
        "rolling": rolling,
    }

    out_path = project_dir / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
