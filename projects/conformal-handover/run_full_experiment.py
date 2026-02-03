"""
Full experiment suite for the paper.

Includes:
- Multiple scenarios (easy/medium/hard)
- Top-K baseline comparison
- Alpha sweep
- Conditional coverage analysis
- Handover-specific metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import json

from src.handover.synthetic_data import generate_dataset, NetworkConfig, MobilityConfig
from src.handover.predictor import train_predictor, get_softmax_scores
from src.handover.conformal import (
    calibrate_threshold,
    predict_sets,
    evaluate_cp,
    run_aci_online,
)


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
        name="Easy (9 cells, low noise)",
        n_gnb_x=3, n_gnb_y=3,
        cell_radius=200.0, noise_std_db=4.0,
        prediction_horizon=5, measurement_noise_db=2.0,
    ),
    "medium": ScenarioConfig(
        name="Medium (16 cells, moderate noise)",
        n_gnb_x=4, n_gnb_y=4,
        cell_radius=150.0, noise_std_db=6.0,
        prediction_horizon=10, measurement_noise_db=4.0,
    ),
    "hard": ScenarioConfig(
        name="Hard (25 cells, high noise)",
        n_gnb_x=5, n_gnb_y=5,
        cell_radius=120.0, noise_std_db=8.0,
        prediction_horizon=15, measurement_noise_db=6.0,
    ),
}


def topk_coverage(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk_preds = np.argsort(probs, axis=1)[:, -k:]
    covered = [labels[i] in topk_preds[i] for i in range(len(labels))]
    return np.mean(covered)


def run_scenario(scenario_key: str, seed: int = 42, n_trajectories: int = 1000):
    """Run full experiment on one scenario."""
    cfg = SCENARIOS[scenario_key]
    print(f"\n{'='*60}")
    print(f"Scenario: {cfg.name}")
    print(f"{'='*60}")

    network_config = NetworkConfig(
        n_gnb_x=cfg.n_gnb_x,
        n_gnb_y=cfg.n_gnb_y,
        cell_radius=cfg.cell_radius,
        noise_std_db=cfg.noise_std_db,
    )
    mobility_config = MobilityConfig(n_trajectories=n_trajectories, trajectory_length=100)

    data = generate_dataset(
        network_config, mobility_config, seed=seed,
        prediction_horizon=cfg.prediction_horizon,
        measurement_noise_db=cfg.measurement_noise_db,
    )

    n_cells = data["n_cells"]
    print(f"Cells: {n_cells}, Samples: {len(data['rsrp'])}")

    traj_ids = data["trajectory_id"]
    n_traj = n_trajectories
    train_traj = set(range(int(0.6 * n_traj)))
    cal_traj = set(range(int(0.6 * n_traj), int(0.8 * n_traj)))
    test_traj = set(range(int(0.8 * n_traj), n_traj))

    train_idx = np.array([i for i, t in enumerate(traj_ids) if t in train_traj])
    cal_idx = np.array([i for i, t in enumerate(traj_ids) if t in cal_traj])
    test_idx = np.array([i for i, t in enumerate(traj_ids) if t in test_traj])

    print(f"Train: {len(train_idx)}, Cal: {len(cal_idx)}, Test: {len(test_idx)}")

    print("Training predictor...")
    model = train_predictor(data, train_idx, cal_idx, n_epochs=20, batch_size=512)

    cal_probs = get_softmax_scores(model, data, cal_idx)
    test_probs = get_softmax_scores(model, data, test_idx)
    cal_labels = data["optimal_cell"][cal_idx]
    test_labels = data["optimal_cell"][test_idx]

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    print(f"Top-1 Accuracy: {top1_acc:.4f}")

    results = {"scenario": cfg.name, "n_cells": n_cells, "top1_acc": top1_acc}

    print("\n--- Top-K Baseline ---")
    for k in [1, 2, 3, 5]:
        cov = topk_coverage(test_probs, test_labels, k)
        results[f"top{k}_coverage"] = cov
        print(f"Top-{k}: coverage={cov:.4f}, size={k}")

    print("\n--- Conformal Prediction (alpha sweep) ---")
    results["cp"] = {}
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        threshold = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
        pred_sets = predict_sets(test_probs, threshold)
        cp_results = evaluate_cp(pred_sets, test_labels)
        results["cp"][f"alpha_{alpha}"] = cp_results
        print(f"CP (α={alpha}): coverage={cp_results['coverage']:.4f}, "
              f"size={cp_results['avg_set_size']:.3f}")

    print("\n--- Adaptive Conformal Inference ---")
    cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]

    test_traj_sorted = sorted(test_traj)
    test_idx_by_traj = []
    for t in test_traj_sorted:
        mask = data["trajectory_id"][test_idx] == t
        test_idx_by_traj.extend(np.where(mask)[0])
    test_idx_by_traj = np.array(test_idx_by_traj)

    test_probs_seq = test_probs[test_idx_by_traj]
    test_labels_seq = test_labels[test_idx_by_traj]

    aci_results = run_aci_online(
        test_probs_seq, test_labels_seq, cal_scores, alpha=0.10, gamma=0.01
    )
    results["aci"] = {
        "coverage": aci_results["coverage"],
        "avg_set_size": aci_results["avg_set_size"],
    }
    print(f"ACI (α=0.10): coverage={aci_results['coverage']:.4f}, "
          f"size={aci_results['avg_set_size']:.3f}")

    print("\n--- Conditional Coverage by Speed ---")
    test_speeds = data["ue_speed"][test_idx]
    speed_quartiles = np.percentile(test_speeds, [25, 50, 75])

    threshold_90 = calibrate_threshold(cal_probs, cal_labels, alpha=0.10)
    pred_sets_90 = predict_sets(test_probs, threshold_90)

    results["conditional"] = {}
    bin_names = ["slow", "medium_slow", "medium_fast", "fast"]
    for i, (low, high) in enumerate([
        (0, speed_quartiles[0]),
        (speed_quartiles[0], speed_quartiles[1]),
        (speed_quartiles[1], speed_quartiles[2]),
        (speed_quartiles[2], np.inf),
    ]):
        mask = (test_speeds >= low) & (test_speeds < high)
        cov = np.mean([test_labels[j] in pred_sets_90[j] for j in np.where(mask)[0]])
        sizes = [len(pred_sets_90[j]) for j in np.where(mask)[0]]
        results["conditional"][bin_names[i]] = {
            "coverage": cov,
            "avg_size": np.mean(sizes),
            "count": int(mask.sum()),
        }
        print(f"  {bin_names[i]}: coverage={cov:.3f}, size={np.mean(sizes):.2f}")

    return results, data, test_idx, pred_sets_90, aci_results


def create_comparison_table(all_results: dict):
    """Create LaTeX table comparing scenarios."""
    print("\n" + "="*60)
    print("LaTeX Table: Top-K vs CP Comparison")
    print("="*60)

    print(r"\begin{table}[!t]")
    print(r"\caption{Top-K vs Conformal Prediction Across Scenarios}")
    print(r"\label{tab:topk-vs-cp}")
    print(r"\centering")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Scenario & Method & Coverage & Avg Size & Adaptive \\")
    print(r"\midrule")

    for scenario_key in ["easy", "medium", "hard"]:
        r = all_results[scenario_key]
        scenario_name = scenario_key.capitalize()

        print(f"\\multirow{{5}}{{*}}{{{scenario_name}}}")
        print(f"& Top-1 & {r['top1_acc']:.3f} & 1 & No \\\\")
        print(f"& Top-2 & {r['top2_coverage']:.3f} & 2 & No \\\\")
        print(f"& Top-3 & {r['top3_coverage']:.3f} & 3 & No \\\\")

        cp = r['cp']['alpha_0.1']
        print(f"& CP ($\\alpha$=0.1) & {cp['coverage']:.3f} & "
              f"\\textbf{{{cp['avg_set_size']:.2f}}} & Yes \\\\")

        aci = r['aci']
        print(f"& ACI ($\\alpha$=0.1) & {aci['coverage']:.3f} & "
              f"{aci['avg_set_size']:.2f} & Yes \\\\")

        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def create_figures(all_results: dict, figures_dir: Path):
    """Create publication-quality figures."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    scenarios = ["easy", "medium", "hard"]
    x = np.arange(len(scenarios))
    width = 0.2

    top1 = [all_results[s]["top1_acc"] for s in scenarios]
    top3 = [all_results[s]["top3_coverage"] for s in scenarios]
    cp_cov = [all_results[s]["cp"]["alpha_0.1"]["coverage"] for s in scenarios]
    cp_size = [all_results[s]["cp"]["alpha_0.1"]["avg_set_size"] for s in scenarios]

    axes[0].bar(x - width, top1, width, label="Top-1", color="C0")
    axes[0].bar(x, top3, width, label="Top-3", color="C1")
    axes[0].bar(x + width, cp_cov, width, label="CP (α=0.1)", color="C2")
    axes[0].axhline(0.9, color="red", linestyle="--", label="Target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Easy", "Medium", "Hard"])
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Coverage by Method")
    axes[0].legend(loc="lower left")
    axes[0].set_ylim(0.5, 1.0)

    axes[1].bar(scenarios, cp_size, color="C2")
    axes[1].set_ylabel("Average Set Size")
    axes[1].set_title("CP Set Size by Scenario")
    for i, v in enumerate(cp_size):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center")

    for scenario in scenarios:
        cond = all_results[scenario]["conditional"]
        coverages = [cond[b]["coverage"] for b in ["slow", "medium_slow", "medium_fast", "fast"]]
        axes[2].plot(["Slow", "Med-S", "Med-F", "Fast"], coverages,
                     marker="o", label=scenario.capitalize())
    axes[2].axhline(0.9, color="red", linestyle="--", label="Target")
    axes[2].set_ylabel("Coverage")
    axes[2].set_title("Conditional Coverage by Speed")
    axes[2].legend()
    axes[2].set_ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig(figures_dir / "scenario_comparison.png", dpi=150)
    print(f"Saved: {figures_dir / 'scenario_comparison.png'}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

    alphas = [0.05, 0.10, 0.15, 0.20]
    for scenario in scenarios:
        coverages = [all_results[scenario]["cp"][f"alpha_{a}"]["coverage"] for a in alphas]
        sizes = [all_results[scenario]["cp"][f"alpha_{a}"]["avg_set_size"] for a in alphas]
        axes2[0].plot([1-a for a in alphas], coverages, marker="o", label=scenario.capitalize())
        axes2[1].plot([1-a for a in alphas], sizes, marker="o", label=scenario.capitalize())

    axes2[0].plot([0.8, 0.95], [0.8, 0.95], "k--", label="Ideal")
    axes2[0].set_xlabel("Target Coverage (1-α)")
    axes2[0].set_ylabel("Empirical Coverage")
    axes2[0].set_title("CP Calibration")
    axes2[0].legend()

    axes2[1].set_xlabel("Target Coverage (1-α)")
    axes2[1].set_ylabel("Average Set Size")
    axes2[1].set_title("Coverage-Size Tradeoff")
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "alpha_sweep.png", dpi=150)
    print(f"Saved: {figures_dir / 'alpha_sweep.png'}")


def main():
    print("="*60)
    print("FULL EXPERIMENT: Conformal Prediction for Handover")
    print("="*60)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    all_results = {}

    for scenario_key in ["easy", "medium", "hard"]:
        results, _, _, _, _ = run_scenario(scenario_key, seed=42, n_trajectories=500)
        all_results[scenario_key] = results

    with open(figures_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved results to {figures_dir / 'results.json'}")

    create_comparison_table(all_results)
    create_figures(all_results, figures_dir)

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    for scenario in ["easy", "medium", "hard"]:
        r = all_results[scenario]
        cp = r["cp"]["alpha_0.1"]
        print(f"\n{scenario.upper()}:")
        print(f"  Top-1 acc: {r['top1_acc']:.3f}")
        print(f"  CP: coverage={cp['coverage']:.3f}, size={cp['avg_set_size']:.2f}")

        if cp["avg_set_size"] < 1.5:
            print(f"  → CP adds minimal value (model already confident)")
        elif cp["avg_set_size"] < 2.5:
            print(f"  → CP provides useful uncertainty quantification")
        else:
            print(f"  → CP essential for reliability (model uncertain)")


if __name__ == "__main__":
    main()
