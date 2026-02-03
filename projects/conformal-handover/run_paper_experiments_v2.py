"""
Paper experiments v2: With confidence intervals and end-to-end metrics.

Improvements over v1:
- Multiple seeds (5) for statistical significance
- End-to-end handover metrics: success rate, measurement overhead
- Error bars on figures
- ± intervals in tables
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import json
from typing import List, Dict
import torch

from src.handover.synthetic_data import generate_dataset, NetworkConfig, MobilityConfig
from src.handover.predictor import train_predictor, get_softmax_scores
from src.handover.conformal import (
    calibrate_threshold,
    predict_sets,
    evaluate_cp,
    run_aci_online,
    GroupConditionalCP,
    assign_speed_groups,
)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


@dataclass
class ScenarioConfig:
    name: str
    n_gnb_x: int
    n_gnb_y: int
    cell_radius: float
    noise_std_db: float
    prediction_horizon: int
    measurement_noise_db: float


SCENARIOS = {
    "easy": ScenarioConfig(
        name="Easy", n_gnb_x=3, n_gnb_y=3,
        cell_radius=200.0, noise_std_db=4.0,
        prediction_horizon=5, measurement_noise_db=2.0,
    ),
    "medium": ScenarioConfig(
        name="Medium", n_gnb_x=4, n_gnb_y=4,
        cell_radius=150.0, noise_std_db=6.0,
        prediction_horizon=10, measurement_noise_db=4.0,
    ),
    "hard": ScenarioConfig(
        name="Hard", n_gnb_x=5, n_gnb_y=5,
        cell_radius=120.0, noise_std_db=8.0,
        prediction_horizon=15, measurement_noise_db=6.0,
    ),
}


def simulate_handover_protocol(
    pred_sets: List[set],
    true_labels: np.ndarray,
    n_cells: int,
    k_max: int = 5,
) -> Dict:
    """
    Simulate actual handover decisions and compute end-to-end metrics.

    Protocol:
    - If |C(x)| <= k_max: predictive handover (measure only cells in set)
    - Else: measurement-based handover (measure all cells)

    Returns:
        success_rate: fraction where true target was in measured set
        measurement_overhead: avg fraction of cells measured (vs exhaustive)
        predictive_rate: fraction using predictive (vs measurement-based)
        rlf_rate: radio link failure rate (true target not in set AND set was small)
    """
    n = len(pred_sets)

    successes = 0
    measurements_total = 0
    predictive_count = 0
    rlf_count = 0

    for i in range(n):
        pred_set = pred_sets[i]
        true_target = true_labels[i]
        set_size = len(pred_set)

        if set_size <= k_max:
            predictive_count += 1
            measurements_total += set_size
            if true_target in pred_set:
                successes += 1
            else:
                rlf_count += 1
        else:
            measurements_total += n_cells
            successes += 1

    return {
        "success_rate": successes / n,
        "measurement_overhead": measurements_total / (n * n_cells),
        "predictive_rate": predictive_count / n,
        "rlf_rate": rlf_count / n,
        "avg_measurements": measurements_total / n,
    }


def run_single_seed(scenario_key: str, seed: int, n_traj: int = 800) -> Dict:
    """Run experiment for one seed, return all metrics."""
    cfg = SCENARIOS[scenario_key]

    np.random.seed(seed)
    torch.manual_seed(seed)

    network_config = NetworkConfig(
        n_gnb_x=cfg.n_gnb_x, n_gnb_y=cfg.n_gnb_y,
        cell_radius=cfg.cell_radius, noise_std_db=cfg.noise_std_db,
    )
    mobility_config = MobilityConfig(n_trajectories=n_traj, trajectory_length=100)

    data = generate_dataset(
        network_config, mobility_config, seed=seed,
        prediction_horizon=cfg.prediction_horizon,
        measurement_noise_db=cfg.measurement_noise_db,
    )

    n_cells = data["n_cells"]
    traj_ids = data["trajectory_id"]

    train_traj = set(range(int(0.6 * n_traj)))
    cal_traj = set(range(int(0.6 * n_traj), int(0.8 * n_traj)))
    test_traj = set(range(int(0.8 * n_traj), n_traj))

    train_idx = np.array([i for i, t in enumerate(traj_ids) if t in train_traj])
    cal_idx = np.array([i for i, t in enumerate(traj_ids) if t in cal_traj])
    test_idx = np.array([i for i, t in enumerate(traj_ids) if t in test_traj])

    model = train_predictor(data, train_idx, cal_idx, n_epochs=20, batch_size=512)

    cal_probs = get_softmax_scores(model, data, cal_idx)
    test_probs = get_softmax_scores(model, data, test_idx)
    cal_labels = data["optimal_cell"][cal_idx]
    test_labels = data["optimal_cell"][test_idx]
    cal_speeds = data["ue_speed"][cal_idx]
    test_speeds = data["ue_speed"][test_idx]

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()

    results = {
        "n_cells": n_cells,
        "top1_acc": float(top1_acc),
    }

    for k in [1, 3, 5]:
        topk = np.argsort(test_probs, axis=1)[:, -k:]
        cov = np.mean([test_labels[i] in topk[i] for i in range(len(test_labels))])
        results[f"top{k}"] = float(cov)

    alpha = 0.10
    threshold = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    pred_sets_std = predict_sets(test_probs, threshold)
    std_eval = evaluate_cp(pred_sets_std, test_labels)
    results["std_coverage"] = std_eval["coverage"]
    results["std_size"] = std_eval["avg_set_size"]

    std_ho = simulate_handover_protocol(pred_sets_std, test_labels, n_cells, k_max=5)
    results["std_success_rate"] = std_ho["success_rate"]
    results["std_measurement_overhead"] = std_ho["measurement_overhead"]
    results["std_predictive_rate"] = std_ho["predictive_rate"]
    results["std_rlf_rate"] = std_ho["rlf_rate"]

    n_groups = 4
    cal_groups = assign_speed_groups(cal_speeds, n_groups)
    test_groups = assign_speed_groups(test_speeds, n_groups)

    gcp = GroupConditionalCP(n_groups=n_groups, alpha=alpha)
    gcp.calibrate(cal_probs, cal_labels, cal_groups)
    pred_sets_gcp = gcp.predict(test_probs, test_groups)
    gcp_eval = evaluate_cp(pred_sets_gcp, test_labels)
    results["gcp_coverage"] = gcp_eval["coverage"]
    results["gcp_size"] = gcp_eval["avg_set_size"]

    gcp_ho = simulate_handover_protocol(pred_sets_gcp, test_labels, n_cells, k_max=5)
    results["gcp_success_rate"] = gcp_ho["success_rate"]
    results["gcp_measurement_overhead"] = gcp_ho["measurement_overhead"]
    results["gcp_predictive_rate"] = gcp_ho["predictive_rate"]
    results["gcp_rlf_rate"] = gcp_ho["rlf_rate"]

    cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    test_traj_sorted = sorted(test_traj)
    test_idx_by_traj = []
    for t in test_traj_sorted:
        mask = data["trajectory_id"][test_idx] == t
        test_idx_by_traj.extend(np.where(mask)[0])
    test_idx_by_traj = np.array(test_idx_by_traj)

    aci_out = run_aci_online(
        test_probs[test_idx_by_traj],
        test_labels[test_idx_by_traj],
        cal_scores, alpha=alpha, gamma=0.01
    )
    results["aci_coverage"] = aci_out["coverage"]
    results["aci_size"] = aci_out["avg_set_size"]

    group_names = ["Slow", "Med-Slow", "Med-Fast", "Fast"]
    for g in range(n_groups):
        mask = test_groups == g
        idxs = np.where(mask)[0]
        std_cov = np.mean([test_labels[i] in pred_sets_std[i] for i in idxs])
        gcp_cov = np.mean([test_labels[i] in pred_sets_gcp[i] for i in idxs])
        results[f"std_cond_{group_names[g]}"] = std_cov
        results[f"gcp_cond_{group_names[g]}"] = gcp_cov

    return results


def aggregate_results(seed_results: List[Dict]) -> Dict:
    """Aggregate results across seeds: compute mean ± std."""
    keys = seed_results[0].keys()
    agg = {}

    for key in keys:
        if key == "n_cells":
            agg[key] = seed_results[0][key]
        else:
            values = [r[key] for r in seed_results]
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)

    return agg


def run_scenario_with_seeds(scenario_key: str, seeds: List[int], n_traj: int = 800) -> Dict:
    """Run scenario with multiple seeds."""
    cfg = SCENARIOS[scenario_key]
    print(f"\n{'='*60}")
    print(f"  {cfg.name} Scenario ({len(seeds)} seeds)")
    print(f"{'='*60}")

    seed_results = []
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})...", end=" ", flush=True)
        result = run_single_seed(scenario_key, seed, n_traj)
        seed_results.append(result)
        print(f"Top-1: {result['top1_acc']:.3f}, CP: {result['std_coverage']:.3f}")

    agg = aggregate_results(seed_results)
    agg["scenario"] = cfg.name
    agg["seed_results"] = seed_results

    print(f"\n  Aggregated (mean ± std):")
    print(f"    Top-1: {agg['top1_acc_mean']:.3f} ± {agg['top1_acc_std']:.3f}")
    print(f"    Std CP: {agg['std_coverage_mean']:.3f} ± {agg['std_coverage_std']:.3f} (size {agg['std_size_mean']:.2f})")
    print(f"    GCP: {agg['gcp_coverage_mean']:.3f} ± {agg['gcp_coverage_std']:.3f}")
    print(f"    ACI: {agg['aci_coverage_mean']:.3f} ± {agg['aci_coverage_std']:.3f}")
    print(f"\n  End-to-End Metrics:")
    print(f"    HO Success Rate: {agg['std_success_rate_mean']:.3f} ± {agg['std_success_rate_std']:.3f}")
    print(f"    Measurement Overhead: {agg['std_measurement_overhead_mean']:.1%} ± {agg['std_measurement_overhead_std']:.1%}")
    print(f"    Predictive HO Rate: {agg['std_predictive_rate_mean']:.1%}")
    print(f"    RLF Rate: {agg['std_rlf_rate_mean']:.3f} ± {agg['std_rlf_rate_std']:.3f}")

    return agg


def create_paper_figures_v2(all_results: Dict, figures_dir: Path):
    """Create figures with error bars."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    scenarios = ["easy", "medium", "hard"]
    x = np.arange(3)
    width = 0.25

    std_cov = [all_results[s]["std_coverage_mean"] for s in scenarios]
    std_cov_err = [all_results[s]["std_coverage_std"] for s in scenarios]
    gcp_cov = [all_results[s]["gcp_coverage_mean"] for s in scenarios]
    gcp_cov_err = [all_results[s]["gcp_coverage_std"] for s in scenarios]
    aci_cov = [all_results[s]["aci_coverage_mean"] for s in scenarios]
    aci_cov_err = [all_results[s]["aci_coverage_std"] for s in scenarios]

    ax = axes[0, 0]
    ax.bar(x - width, std_cov, width, yerr=std_cov_err, label="Standard CP", color="C0", capsize=3)
    ax.bar(x, gcp_cov, width, yerr=gcp_cov_err, label="Group-Cond CP", color="C1", capsize=3)
    ax.bar(x + width, aci_cov, width, yerr=aci_cov_err, label="ACI", color="C2", capsize=3)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5, label="Target (90%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Coverage")
    ax.set_title("(a) Coverage (mean ± std, 5 seeds)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0.82, 0.98)

    std_size = [all_results[s]["std_size_mean"] for s in scenarios]
    std_size_err = [all_results[s]["std_size_std"] for s in scenarios]
    gcp_size = [all_results[s]["gcp_size_mean"] for s in scenarios]
    gcp_size_err = [all_results[s]["gcp_size_std"] for s in scenarios]

    ax = axes[0, 1]
    ax.bar(x - width/2, std_size, width, yerr=std_size_err, label="Standard CP", color="C0", capsize=3)
    ax.bar(x + width/2, gcp_size, width, yerr=gcp_size_err, label="Group-Cond CP", color="C1", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Average Set Size")
    ax.set_title("(b) Prediction Set Size")
    ax.legend()

    ax = axes[1, 0]
    success = [all_results[s]["std_success_rate_mean"] for s in scenarios]
    success_err = [all_results[s]["std_success_rate_std"] for s in scenarios]
    overhead = [all_results[s]["std_measurement_overhead_mean"] for s in scenarios]
    overhead_err = [all_results[s]["std_measurement_overhead_std"] for s in scenarios]

    ax.bar(x - width/2, success, width, yerr=success_err, label="HO Success Rate", color="C0", capsize=3)
    ax.bar(x + width/2, overhead, width, yerr=overhead_err, label="Meas. Overhead", color="C3", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Rate")
    ax.set_title("(c) End-to-End: Success Rate & Measurement Overhead")
    ax.legend()
    ax.set_ylim(0, 1.1)

    ax = axes[1, 1]
    rlf = [all_results[s]["std_rlf_rate_mean"] * 100 for s in scenarios]
    rlf_err = [all_results[s]["std_rlf_rate_std"] * 100 for s in scenarios]
    pred_rate = [all_results[s]["std_predictive_rate_mean"] * 100 for s in scenarios]

    ax.bar(x - width/2, rlf, width, yerr=rlf_err, label="RLF Rate (%)", color="C3", capsize=3)
    ax.bar(x + width/2, pred_rate, width, label="Predictive HO (%)", color="C2", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Percentage")
    ax.set_title("(d) RLF Rate & Predictive Handover Usage")
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "main_results_v2.png", dpi=200, bbox_inches="tight")
    plt.savefig(figures_dir / "main_results_v2.pdf", bbox_inches="tight")
    print(f"Saved: main_results_v2.png/pdf")


def generate_latex_tables_v2(all_results: Dict):
    """Generate LaTeX tables with confidence intervals."""

    print("\n" + "="*70)
    print("TABLE 1: Main Results with Confidence Intervals")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{CP Results Across Scenarios ($\alpha=0.1$, mean $\pm$ std over 5 seeds)}
\label{tab:main}
\centering
\begin{tabular}{llcc}
\toprule
Scenario & Method & Coverage & Avg Size \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        n_cells = r["n_cells"]
        print(f"\\multirow{{4}}{{*}}{{{s.capitalize()} ({n_cells} cells)}}")
        print(f"  & Top-1 & ${r['top1_acc_mean']:.3f} \\pm {r['top1_acc_std']:.3f}$ & 1.00 \\\\")
        print(f"  & Top-3 & ${r['top3_mean']:.3f} \\pm {r['top3_std']:.3f}$ & 3.00 \\\\")
        print(f"  & Standard CP & ${r['std_coverage_mean']:.3f} \\pm {r['std_coverage_std']:.3f}$ & ${r['std_size_mean']:.2f} \\pm {r['std_size_std']:.2f}$ \\\\")
        print(f"  & ACI & ${r['aci_coverage_mean']:.3f} \\pm {r['aci_coverage_std']:.3f}$ & ${r['aci_size_mean']:.2f} \\pm {r['aci_size_std']:.2f}$ \\\\")
        if s != "hard":
            print("\\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*70)
    print("TABLE 2: End-to-End Handover Metrics")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{End-to-End Handover Performance with CP ($K_{\max}=5$)}
\label{tab:e2e}
\centering
\begin{tabular}{lccc}
\toprule
Scenario & HO Success & Meas. Overhead & RLF Rate \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"{s.capitalize()} & "
              f"${r['std_success_rate_mean']:.3f} \\pm {r['std_success_rate_std']:.3f}$ & "
              f"${r['std_measurement_overhead_mean']*100:.1f}\\% \\pm {r['std_measurement_overhead_std']*100:.1f}\\%$ & "
              f"${r['std_rlf_rate_mean']*100:.2f}\\%$ \\\\")

    print(r"""\midrule
\multicolumn{4}{l}{\small Exhaustive (baseline): Success=100\%, Overhead=100\%, RLF=0\%} \\
\bottomrule
\end{tabular}
\end{table}""")


def main():
    print("="*70)
    print("  PAPER EXPERIMENTS v2: With Confidence Intervals & E2E Metrics")
    print("="*70)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    seeds = [42, 123, 456, 789, 1011]

    all_results = {}
    for scenario in ["easy", "medium", "hard"]:
        all_results[scenario] = run_scenario_with_seeds(scenario, seeds, n_traj=600)

    with open(figures_dir / "paper_results_v2.json", "w") as f:
        serializable = {}
        for s, r in all_results.items():
            serializable[s] = {k: v for k, v in r.items() if k != "seed_results"}
        json.dump(serializable, f, indent=2)

    create_paper_figures_v2(all_results, figures_dir)
    generate_latex_tables_v2(all_results)

    print("\n" + "="*70)
    print("  KEY INSIGHTS")
    print("="*70)

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"\n{s.upper()}:")
        print(f"  Coverage: {r['std_coverage_mean']:.1%} ± {r['std_coverage_std']:.1%}")
        print(f"  HO Success: {r['std_success_rate_mean']:.1%} (vs 100% exhaustive)")
        print(f"  Measurement savings: {(1-r['std_measurement_overhead_mean'])*100:.0f}%")
        print(f"  RLF rate: {r['std_rlf_rate_mean']*100:.2f}%")


if __name__ == "__main__":
    main()
