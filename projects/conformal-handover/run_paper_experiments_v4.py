"""
Paper experiments v4: With 3dB baseline and ping-pong analysis.

Changes from v3:
- Add 3dB threshold baseline comparison
- Add ping-pong handover tracking
- Add UE mobility visualization
- Generate updated LaTeX tables
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from pathlib import Path
from dataclasses import dataclass
import json
from typing import List, Dict
import torch

from src.handover.synthetic_data import (
    generate_dataset,
    NetworkConfig,
    MobilityConfig,
    generate_gnb_positions,
    generate_trajectory,
)
from src.handover.predictor import train_predictor, get_softmax_scores
from src.handover.conformal import (
    calibrate_threshold,
    predict_sets,
    evaluate_cp,
    run_aci_online,
    calibrate_raps,
    predict_sets_raps,
)
from src.handover.baseline import (
    predict_3db_baseline,
    evaluate_3db_baseline,
    simulate_3db_handover_sequence,
    simulate_policy_serving_sequence,
    compute_ping_pong_metrics,
    count_ping_pong,
    count_handovers,
)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
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
    """Simulate handover decisions and compute end-to-end metrics."""
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
    }


def compute_ping_pong_by_policy(
    data: dict,
    test_idx: np.ndarray,
    ml_predictions: np.ndarray,
    prediction_sets: list,
    k_max: int = 5,
) -> Dict:
    """
    Compute ping-pong metrics for each policy by simulating serving sequences.

    Policies:
    - 3dB: reactive handover when neighbor >3dB stronger
    - ML Top-1: always switch to ML prediction
    - CP Adaptive: trust ML when confident, stay if serving in set
    """
    traj_ids = data["trajectory_id"][test_idx]
    rsrp = data["rsrp"][test_idx]
    initial_serving = data["serving_cell"][test_idx]

    unique_trajs = np.unique(traj_ids)

    metrics = {
        "3db": {"ho": 0, "pp": 0, "steps": 0},
        "ml_top1": {"ho": 0, "pp": 0, "steps": 0},
        "cp_adaptive": {"ho": 0, "pp": 0, "steps": 0},
    }

    for traj in unique_trajs:
        mask = traj_ids == traj
        traj_rsrp = rsrp[mask]
        traj_ml_pred = ml_predictions[mask]
        traj_pred_sets = [prediction_sets[i] for i in np.where(mask)[0]]
        init_serving = initial_serving[mask][0]

        for policy in ["3db", "ml_top1", "cp_adaptive"]:
            serving_seq = simulate_policy_serving_sequence(
                traj_rsrp,
                init_serving,
                policy=policy,
                ml_predictions=traj_ml_pred,
                prediction_sets=traj_pred_sets,
                threshold_db=3.0,
                k_max=k_max,
            )
            metrics[policy]["ho"] += count_handovers(serving_seq)
            metrics[policy]["pp"] += count_ping_pong(serving_seq)
            metrics[policy]["steps"] += len(serving_seq)

    results = {}
    for policy in ["3db", "ml_top1", "cp_adaptive"]:
        m = metrics[policy]
        pp_rate = m["pp"] / max(m["ho"], 1)
        ho_rate = m["ho"] / m["steps"]
        results[f"{policy}_ho_count"] = m["ho"]
        results[f"{policy}_pp_count"] = m["pp"]
        results[f"{policy}_pp_rate"] = pp_rate
        results[f"{policy}_ho_rate"] = ho_rate

    return results


def run_single_seed(scenario_key: str, seed: int, n_traj: int = 600) -> Dict:
    """Run experiment for one seed."""
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

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()

    results = {"n_cells": n_cells, "top1_acc": float(top1_acc)}

    for k in [1, 3, 5]:
        topk = np.argsort(test_probs, axis=1)[:, -k:]
        cov = np.mean([test_labels[i] in topk[i] for i in range(len(test_labels))])
        results[f"top{k}"] = float(cov)

    alpha = 0.10

    threshold_std = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    pred_sets_std = predict_sets(test_probs, threshold_std)
    std_eval = evaluate_cp(pred_sets_std, test_labels)
    results["std_coverage"] = std_eval["coverage"]
    results["std_size"] = std_eval["avg_set_size"]

    std_ho = simulate_handover_protocol(pred_sets_std, test_labels, n_cells, k_max=5)
    results["std_success_rate"] = std_ho["success_rate"]
    results["std_measurement_overhead"] = std_ho["measurement_overhead"]
    results["std_rlf_rate"] = std_ho["rlf_rate"]

    threshold_raps = calibrate_raps(cal_probs, cal_labels, alpha=alpha, k_reg=1, lam=0.01)
    pred_sets_raps_list = predict_sets_raps(test_probs, threshold_raps, k_reg=1, lam=0.01)
    raps_eval = evaluate_cp(pred_sets_raps_list, test_labels)
    results["raps_coverage"] = raps_eval["coverage"]
    results["raps_size"] = raps_eval["avg_set_size"]

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

    test_rsrp = data["rsrp"][test_idx]
    test_serving = data["serving_cell"][test_idx]
    baseline_eval = evaluate_3db_baseline(test_rsrp, test_serving, test_labels, threshold_db=3.0)
    results["baseline_3db_acc"] = baseline_eval["accuracy"]
    results["baseline_3db_ho_rate"] = baseline_eval["ho_rate"]
    results["baseline_3db_missed_ho"] = baseline_eval["missed_ho_rate"]
    results["baseline_3db_unnecessary_ho"] = baseline_eval["unnecessary_ho_rate"]

    ml_predictions = test_probs.argmax(axis=1)
    pp_metrics = compute_ping_pong_by_policy(
        data, test_idx, ml_predictions, pred_sets_std, k_max=5
    )
    results["3db_pp_rate"] = pp_metrics["3db_pp_rate"]
    results["3db_ho_rate"] = pp_metrics["3db_ho_rate"]
    results["ml_top1_pp_rate"] = pp_metrics["ml_top1_pp_rate"]
    results["ml_top1_ho_rate"] = pp_metrics["ml_top1_ho_rate"]
    results["cp_adaptive_pp_rate"] = pp_metrics["cp_adaptive_pp_rate"]
    results["cp_adaptive_ho_rate"] = pp_metrics["cp_adaptive_ho_rate"]

    return results


def aggregate_results(seed_results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    keys = seed_results[0].keys()
    agg = {}
    for key in keys:
        if key in ["n_cells", "total_handovers"]:
            agg[key] = seed_results[0][key]
        else:
            values = [r[key] for r in seed_results]
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
    return agg


def run_scenario_with_seeds(scenario_key: str, seeds: List[int], n_traj: int = 600) -> Dict:
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
        print(f"Top-1: {result['top1_acc']:.3f}, 3dB: {result['baseline_3db_acc']:.3f}, CP: {result['std_coverage']:.3f}")

    agg = aggregate_results(seed_results)
    agg["scenario"] = cfg.name

    print(f"\n  Summary:")
    print(f"    3dB Baseline: {agg['baseline_3db_acc_mean']:.3f} ± {agg['baseline_3db_acc_std']:.3f}")
    print(f"    ML Top-1: {agg['top1_acc_mean']:.3f} ± {agg['top1_acc_std']:.3f}")
    print(f"    Standard CP: {agg['std_coverage_mean']:.3f} ± {agg['std_coverage_std']:.3f} (size {agg['std_size_mean']:.2f})")
    print(f"    Ping-pong rates: 3dB={agg['3db_pp_rate_mean']:.2f}, ML={agg['ml_top1_pp_rate_mean']:.2f}, CP={agg['cp_adaptive_pp_rate_mean']:.2f}")

    return agg


def create_ue_mobility_figure(figures_dir: Path):
    """Create UE mobility visualization figure."""
    np.random.seed(42)

    network_config = NetworkConfig(n_gnb_x=4, n_gnb_y=4, cell_radius=150.0)
    gnb_positions = generate_gnb_positions(network_config)
    mobility_config = MobilityConfig(n_trajectories=5, trajectory_length=100)
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(8, 7))

    cell_radius = network_config.cell_radius
    hex_radius = cell_radius * 0.95

    for i, (x, y) in enumerate(gnb_positions):
        hex_patch = RegularPolygon(
            (x, y), numVertices=6, radius=hex_radius,
            orientation=np.pi / 6,
            facecolor='lightblue', edgecolor='steelblue',
            alpha=0.3, linewidth=1.5
        )
        ax.add_patch(hex_patch)

    ax.scatter(gnb_positions[:, 0], gnb_positions[:, 1],
               s=200, c='darkblue', marker='^', zorder=10, label='gNB')

    for i, (x, y) in enumerate(gnb_positions):
        ax.annotate(f'{i+1}', (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for traj_idx in range(5):
        traj = generate_trajectory(gnb_positions, mobility_config, rng)

        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[traj_idx],
                linewidth=2, alpha=0.8, label=f'UE {traj_idx+1}')

        ax.scatter(traj[0, 0], traj[0, 1], s=100, c=colors[traj_idx],
                   marker='o', edgecolor='white', linewidth=2, zorder=11)

        ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c=colors[traj_idx],
                   marker='s', edgecolor='white', linewidth=2, zorder=11)

        for t in [25, 50, 75]:
            if t < len(traj):
                ax.scatter(traj[t, 0], traj[t, 1], s=30, c=colors[traj_idx],
                           marker='o', alpha=0.5, zorder=9)

    x_min, y_min = gnb_positions.min(axis=0) - 150
    x_max, y_max = gnb_positions.max(axis=0) + 150
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    ax.set_title('UE Mobility in 4×4 gNB Grid (Medium Scenario)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "ue_mobility.png", dpi=200, bbox_inches="tight")
    plt.savefig(figures_dir / "ue_mobility.pdf", bbox_inches="tight")
    print(f"Saved: ue_mobility.png/pdf")
    plt.close()


def create_paper_figures_v4(all_results: Dict, figures_dir: Path):
    """Create final publication figures."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    scenarios = ["easy", "medium", "hard"]
    x = np.arange(3)
    width = 0.18

    baseline_acc = [all_results[s]["baseline_3db_acc_mean"] for s in scenarios]
    baseline_err = [all_results[s]["baseline_3db_acc_std"] for s in scenarios]
    top1 = [all_results[s]["top1_acc_mean"] for s in scenarios]
    top1_err = [all_results[s]["top1_acc_std"] for s in scenarios]
    std_cov = [all_results[s]["std_coverage_mean"] for s in scenarios]
    std_cov_err = [all_results[s]["std_coverage_std"] for s in scenarios]

    ax = axes[0, 0]
    ax.bar(x - width, baseline_acc, width, yerr=baseline_err, label="3dB Baseline", color="gray", capsize=3)
    ax.bar(x, top1, width, yerr=top1_err, label="ML Top-1", color="C0", capsize=3)
    ax.bar(x + width, std_cov, width, yerr=std_cov_err, label="ML + CP", color="C2", capsize=3)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5, label="Target (90%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy (9)", "Medium (16)", "Hard (25)"])
    ax.set_ylabel("Accuracy / Coverage")
    ax.set_title("(a) Prediction Accuracy Comparison")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0.3, 1.05)

    std_size = [all_results[s]["std_size_mean"] for s in scenarios]
    std_size_err = [all_results[s]["std_size_std"] for s in scenarios]

    ax = axes[0, 1]
    bars = ax.bar(x, std_size, 0.5, yerr=std_size_err, color="C2", capsize=3)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1, label="3dB (always 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Average Set Size")
    ax.set_title("(b) CP Prediction Set Size")
    ax.legend()

    for i, (bar, size) in enumerate(zip(bars, std_size)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{size:.1f}', ha='center', va='bottom', fontsize=10)

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
    ax.set_title("(c) End-to-End Performance")
    ax.legend()
    ax.set_ylim(0, 1.1)

    ax = axes[1, 1]
    pp_3db = [all_results[s]["3db_pp_rate_mean"] for s in scenarios]
    pp_3db_err = [all_results[s]["3db_pp_rate_std"] for s in scenarios]
    pp_ml = [all_results[s]["ml_top1_pp_rate_mean"] for s in scenarios]
    pp_ml_err = [all_results[s]["ml_top1_pp_rate_std"] for s in scenarios]
    pp_cp = [all_results[s]["cp_adaptive_pp_rate_mean"] for s in scenarios]
    pp_cp_err = [all_results[s]["cp_adaptive_pp_rate_std"] for s in scenarios]

    ax.bar(x - width, pp_3db, width, yerr=pp_3db_err, label="3dB", color="gray", capsize=3)
    ax.bar(x, pp_ml, width, yerr=pp_ml_err, label="ML Top-1", color="C0", capsize=3)
    ax.bar(x + width, pp_cp, width, yerr=pp_cp_err, label="CP Adaptive", color="C2", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Ping-Pong Rate")
    ax.set_title("(d) Ping-Pong Rate by Policy")
    ax.legend(loc="upper right", fontsize=8)
    max_pp = max(max(pp_3db), max(pp_ml), max(pp_cp))
    ax.set_ylim(0, max_pp * 1.3 if max_pp > 0 else 0.1)

    plt.tight_layout()
    plt.savefig(figures_dir / "main_results_v4.png", dpi=200, bbox_inches="tight")
    plt.savefig(figures_dir / "main_results_v4.pdf", bbox_inches="tight")
    print(f"Saved: main_results_v4.png/pdf")


def generate_latex_tables_v4(all_results: Dict):
    """Generate final LaTeX tables."""

    print("\n" + "="*70)
    print("TABLE 1: Main Results with 3dB Baseline (5 seeds)")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{Handover Prediction Results ($\alpha=0.1$, 5 seeds)}
\label{tab:main}
\centering
\begin{tabular}{llccc}
\toprule
Scenario & Method & Coverage & Avg Size & Set Adaptive \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        n_cells = r["n_cells"]
        print(f"\\multirow{{4}}{{*}}{{{s.capitalize()} ({n_cells})}}")
        print(f"  & 3dB Baseline & ${r['baseline_3db_acc_mean']:.2f} \\pm {r['baseline_3db_acc_std']:.2f}$ & 1 & No \\\\")
        print(f"  & ML Top-1 & ${r['top1_acc_mean']:.2f} \\pm {r['top1_acc_std']:.2f}$ & 1 & No \\\\")
        print(f"  & ML Top-3 & ${r['top3_mean']:.2f} \\pm {r['top3_std']:.2f}$ & 3 & No \\\\")
        print(f"  & ML + CP & ${r['std_coverage_mean']:.2f} \\pm {r['std_coverage_std']:.2f}$ & ${r['std_size_mean']:.1f} \\pm {r['std_size_std']:.1f}$ & \\textbf{{Yes}} \\\\")
        if s != "hard":
            print("\\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*70)
    print("TABLE 2: AIML to System-Level KPI Mapping")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{AIML KPI to System-Level Metric Mapping}
\label{tab:kpi}
\centering
\begin{tabular}{lll}
\toprule
AIML Metric & System Metric & Interpretation \\
\midrule
Top-1 Accuracy & HO Success (no CP) & Wrong pred $\rightarrow$ RLF \\
Coverage & HO Success (with CP) & True in set $\rightarrow$ meas. finds it \\
Set Size & Meas. Overhead & Cells to measure before HO \\
Undercoverage & RLF Rate & True not in set $\rightarrow$ dropped \\
\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*70)
    print("TABLE 3: End-to-End Handover Performance")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{End-to-End Handover Performance ($K_{\max}=5$)}
\label{tab:e2e}
\centering
\begin{tabular}{lcccc}
\toprule
Scenario & HO Success & Overhead & Savings & RLF \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        savings = (1 - r['std_measurement_overhead_mean']) * 100
        print(f"{s.capitalize()} & "
              f"${r['std_success_rate_mean']*100:.1f}\\%$ & "
              f"${r['std_measurement_overhead_mean']*100:.1f}\\%$ & "
              f"\\textbf{{{savings:.0f}\\%}} & "
              f"${r['std_rlf_rate_mean']*100:.1f}\\%$ \\\\")

    print(r"""\midrule
Exhaustive & 100\% & 100\% & 0\% & 0\% \\
\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*70)
    print("TABLE 4: Ping-Pong Handover Comparison")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{Ping-Pong Rate by Handover Policy}
\label{tab:pingpong}
\centering
\begin{tabular}{lccc}
\toprule
Scenario & 3dB Baseline & ML Top-1 & CP Adaptive \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"{s.capitalize()} & "
              f"${r['3db_pp_rate_mean']:.2f} \\pm {r['3db_pp_rate_std']:.2f}$ & "
              f"${r['ml_top1_pp_rate_mean']:.2f} \\pm {r['ml_top1_pp_rate_std']:.2f}$ & "
              f"$\\mathbf{{{r['cp_adaptive_pp_rate_mean']:.2f} \\pm {r['cp_adaptive_pp_rate_std']:.2f}}}$ \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    print("="*70)
    print("  PAPER EXPERIMENTS v4: With 3dB Baseline & Ping-Pong Analysis")
    print("="*70)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    print("\nGenerating UE mobility figure...")
    create_ue_mobility_figure(figures_dir)

    seeds = [42, 123, 456, 789, 1011]

    all_results = {}
    for scenario in ["easy", "medium", "hard"]:
        all_results[scenario] = run_scenario_with_seeds(scenario, seeds, n_traj=600)

    with open(figures_dir / "paper_results_v4.json", "w") as f:
        json.dump({s: {k: v for k, v in r.items() if not k.endswith("_results")}
                   for s, r in all_results.items()}, f, indent=2)

    create_paper_figures_v4(all_results, figures_dir)
    generate_latex_tables_v4(all_results)

    print("\n" + "="*70)
    print("  KEY FINDINGS")
    print("="*70)

    print("\n1. 3dB Baseline vs ML:")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        diff = r['top1_acc_mean'] - r['baseline_3db_acc_mean']
        print(f"   {s}: ML improves {diff:+.1%} over 3dB baseline")

    print("\n2. ML + CP vs 3dB Baseline:")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        diff = r['std_coverage_mean'] - r['baseline_3db_acc_mean']
        print(f"   {s}: CP achieves {r['std_coverage_mean']:.1%} vs 3dB's {r['baseline_3db_acc_mean']:.1%} ({diff:+.1%})")

    print("\n3. Measurement Savings:")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        savings = (1 - r['std_measurement_overhead_mean']) * 100
        print(f"   {s}: {savings:.0f}% savings vs exhaustive")

    print("\n4. Ping-Pong Reduction (CP vs 3dB):")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        reduction = (r['3db_pp_rate_mean'] - r['cp_adaptive_pp_rate_mean']) / max(r['3db_pp_rate_mean'], 0.01) * 100
        print(f"   {s}: 3dB={r['3db_pp_rate_mean']:.2f}, CP={r['cp_adaptive_pp_rate_mean']:.2f} ({reduction:+.0f}% reduction)")


if __name__ == "__main__":
    main()
