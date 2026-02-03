"""
Paper experiments: Complete experimental suite.

Includes:
- Multiple scenarios
- Standard CP vs Group-Conditional CP vs ACI
- Conditional coverage analysis
- Publication-quality figures
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


def run_single_scenario(scenario_key: str, seed: int = 42, n_traj: int = 800):
    """Run complete experiment on one scenario."""
    cfg = SCENARIOS[scenario_key]
    print(f"\n{'='*50}")
    print(f"  {cfg.name} Scenario")
    print(f"{'='*50}")

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

    print(f"Cells: {n_cells} | Train: {len(train_idx)} | Cal: {len(cal_idx)} | Test: {len(test_idx)}")

    print("Training model...", end=" ", flush=True)
    model = train_predictor(data, train_idx, cal_idx, n_epochs=20, batch_size=512)
    print("Done")

    cal_probs = get_softmax_scores(model, data, cal_idx)
    test_probs = get_softmax_scores(model, data, test_idx)
    cal_labels = data["optimal_cell"][cal_idx]
    test_labels = data["optimal_cell"][test_idx]
    cal_speeds = data["ue_speed"][cal_idx]
    test_speeds = data["ue_speed"][test_idx]

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    print(f"Top-1 Accuracy: {top1_acc:.3f}")

    results = {
        "scenario": cfg.name,
        "n_cells": n_cells,
        "top1_acc": float(top1_acc),
    }

    for k in [1, 2, 3, 5]:
        topk = np.argsort(test_probs, axis=1)[:, -k:]
        cov = np.mean([test_labels[i] in topk[i] for i in range(len(test_labels))])
        results[f"top{k}"] = float(cov)

    alpha = 0.10
    threshold = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    pred_sets_std = predict_sets(test_probs, threshold)
    std_results = evaluate_cp(pred_sets_std, test_labels)
    results["standard_cp"] = {k: float(v) for k, v in std_results.items()}
    print(f"Standard CP: cov={std_results['coverage']:.3f}, size={std_results['avg_set_size']:.2f}")

    n_groups = 4
    cal_groups = assign_speed_groups(cal_speeds, n_groups)
    test_groups = assign_speed_groups(test_speeds, n_groups)

    gcp = GroupConditionalCP(n_groups=n_groups, alpha=alpha)
    gcp.calibrate(cal_probs, cal_labels, cal_groups)
    pred_sets_gcp = gcp.predict(test_probs, test_groups)
    gcp_results = evaluate_cp(pred_sets_gcp, test_labels)
    results["group_cp"] = {k: float(v) for k, v in gcp_results.items()}
    print(f"Group-Cond CP: cov={gcp_results['coverage']:.3f}, size={gcp_results['avg_set_size']:.2f}")

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
    results["aci"] = {
        "coverage": float(aci_out["coverage"]),
        "avg_set_size": float(aci_out["avg_set_size"]),
    }
    print(f"ACI: cov={aci_out['coverage']:.3f}, size={aci_out['avg_set_size']:.2f}")

    results["conditional"] = {"standard": {}, "group_cp": {}}
    group_names = ["Slow", "Med-Slow", "Med-Fast", "Fast"]

    print("\nConditional Coverage by Speed:")
    print(f"  {'Group':<10} {'Std CP':<12} {'Group-CP':<12}")
    print(f"  {'-'*34}")

    for g in range(n_groups):
        mask = test_groups == g
        idxs = np.where(mask)[0]

        std_cov = np.mean([test_labels[i] in pred_sets_std[i] for i in idxs])
        std_size = np.mean([len(pred_sets_std[i]) for i in idxs])

        gcp_cov = np.mean([test_labels[i] in pred_sets_gcp[i] for i in idxs])
        gcp_size = np.mean([len(pred_sets_gcp[i]) for i in idxs])

        results["conditional"]["standard"][group_names[g]] = {
            "coverage": float(std_cov), "size": float(std_size)
        }
        results["conditional"]["group_cp"][group_names[g]] = {
            "coverage": float(gcp_cov), "size": float(gcp_size)
        }

        std_ok = "✓" if std_cov >= 0.89 else "✗"
        gcp_ok = "✓" if gcp_cov >= 0.89 else "✗"
        print(f"  {group_names[g]:<10} {std_cov:.3f} {std_ok:<5} {gcp_cov:.3f} {gcp_ok}")

    return results, aci_out


def create_paper_figures(all_results: dict, aci_outputs: dict, figures_dir: Path):
    """Create publication-quality figures."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    scenarios = ["easy", "medium", "hard"]
    x = np.arange(3)
    width = 0.22

    std_cov = [all_results[s]["standard_cp"]["coverage"] for s in scenarios]
    gcp_cov = [all_results[s]["group_cp"]["coverage"] for s in scenarios]
    std_size = [all_results[s]["standard_cp"]["avg_set_size"] for s in scenarios]
    gcp_size = [all_results[s]["group_cp"]["avg_set_size"] for s in scenarios]

    ax = axes[0, 0]
    ax.bar(x - width/2, std_cov, width, label="Standard CP", color="C0")
    ax.bar(x + width/2, gcp_cov, width, label="Group-Cond CP", color="C1")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5, label="Target (90%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Marginal Coverage")
    ax.set_title("(a) Marginal Coverage")
    ax.legend(loc="lower left")
    ax.set_ylim(0.85, 0.95)

    ax = axes[0, 1]
    ax.bar(x - width/2, std_size, width, label="Standard CP", color="C0")
    ax.bar(x + width/2, gcp_size, width, label="Group-Cond CP", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Average Set Size")
    ax.set_title("(b) Prediction Set Size")
    ax.legend()

    ax = axes[1, 0]
    group_names = ["Slow", "Med-Slow", "Med-Fast", "Fast"]
    x_groups = np.arange(4)

    for i, scenario in enumerate(scenarios):
        std_cond = [all_results[scenario]["conditional"]["standard"][g]["coverage"]
                    for g in group_names]
        ax.plot(x_groups, std_cond, marker="o", linestyle="--",
                label=f"{scenario.capitalize()} (Std)", alpha=0.7)

    ax.axhline(0.9, color="red", linestyle="-", linewidth=1.5, label="Target")
    ax.set_xticks(x_groups)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Coverage")
    ax.set_title("(c) Conditional Coverage - Standard CP")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.75, 1.0)

    ax = axes[1, 1]
    for i, scenario in enumerate(scenarios):
        gcp_cond = [all_results[scenario]["conditional"]["group_cp"][g]["coverage"]
                    for g in group_names]
        ax.plot(x_groups, gcp_cond, marker="s", linestyle="-",
                label=f"{scenario.capitalize()} (GCP)", alpha=0.9)

    ax.axhline(0.9, color="red", linestyle="-", linewidth=1.5, label="Target")
    ax.set_xticks(x_groups)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Coverage")
    ax.set_title("(d) Conditional Coverage - Group-Conditional CP")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.75, 1.0)

    plt.tight_layout()
    plt.savefig(figures_dir / "main_results.png", dpi=200, bbox_inches="tight")
    plt.savefig(figures_dir / "main_results.pdf", bbox_inches="tight")
    print(f"Saved: main_results.png/pdf")

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.5))

    for i, scenario in enumerate(scenarios):
        aci = aci_outputs[scenario]
        window = 200
        rolling = np.convolve(aci["coverages"], np.ones(window)/window, mode="valid")
        axes2[i].plot(rolling, color="C0", alpha=0.8)
        axes2[i].axhline(0.9, color="red", linestyle="--", label="Target")
        axes2[i].fill_between(range(len(rolling)), 0.85, 0.95, alpha=0.1, color="green")
        axes2[i].set_xlabel("Time Step")
        axes2[i].set_ylabel("Rolling Coverage")
        axes2[i].set_title(f"{scenario.capitalize()} Scenario")
        axes2[i].set_ylim(0.8, 1.0)
        axes2[i].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(figures_dir / "aci_rolling.png", dpi=200, bbox_inches="tight")
    plt.savefig(figures_dir / "aci_rolling.pdf", bbox_inches="tight")
    print(f"Saved: aci_rolling.png/pdf")


def generate_latex_tables(all_results: dict):
    """Generate LaTeX tables for paper."""

    print("\n" + "="*60)
    print("TABLE 1: Main Results")
    print("="*60)
    print(r"""
\begin{table}[t]
\caption{Conformal Prediction Results Across Scenarios ($\alpha=0.1$)}
\label{tab:main}
\centering
\begin{tabular}{llcccc}
\toprule
Scenario & Method & Coverage & Avg Size & Adaptive \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"\\multirow{{4}}{{*}}{{{s.capitalize()}}}")
        print(f"  & Top-1 & {r['top1']:.3f} & 1.00 & No \\\\")
        print(f"  & Top-3 & {r['top3']:.3f} & 3.00 & No \\\\")
        print(f"  & Standard CP & {r['standard_cp']['coverage']:.3f} & {r['standard_cp']['avg_set_size']:.2f} & Yes \\\\")
        print(f"  & Group-Cond CP & {r['group_cp']['coverage']:.3f} & {r['group_cp']['avg_set_size']:.2f} & Yes \\\\")
        if s != "hard":
            print("\\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*60)
    print("TABLE 2: Conditional Coverage Gap Analysis")
    print("="*60)
    print(r"""
\begin{table}[t]
\caption{Conditional Coverage by UE Speed ($\alpha=0.1$)}
\label{tab:conditional}
\centering
\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{4}{c}{Speed Quartile} \\
\cmidrule(lr){3-6}
Scenario & Method & Slow & Med-S & Med-F & Fast \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        std = r["conditional"]["standard"]
        gcp = r["conditional"]["group_cp"]

        def fmt(v):
            return f"\\textbf{{{v:.2f}}}" if v < 0.89 else f"{v:.2f}"

        print(f"\\multirow{{2}}{{*}}{{{s.capitalize()}}}")
        print(f"  & Std CP & {fmt(std['Slow']['coverage'])} & {fmt(std['Med-Slow']['coverage'])} & {fmt(std['Med-Fast']['coverage'])} & {fmt(std['Fast']['coverage'])} \\\\")
        print(f"  & GCP & {gcp['Slow']['coverage']:.2f} & {gcp['Med-Slow']['coverage']:.2f} & {gcp['Med-Fast']['coverage']:.2f} & {gcp['Fast']['coverage']:.2f} \\\\")
        if s != "hard":
            print("\\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    print("="*60)
    print("  PAPER EXPERIMENTS: Conformal Prediction for Handover")
    print("="*60)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    all_results = {}
    aci_outputs = {}

    for scenario in ["easy", "medium", "hard"]:
        results, aci_out = run_single_scenario(scenario, seed=42, n_traj=800)
        all_results[scenario] = results
        aci_outputs[scenario] = aci_out

    with open(figures_dir / "paper_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    create_paper_figures(all_results, aci_outputs, figures_dir)
    generate_latex_tables(all_results)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print("\nKey findings:")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        std = r["standard_cp"]
        gcp = r["group_cp"]

        std_gaps = [r["conditional"]["standard"][g]["coverage"] < 0.89
                    for g in ["Slow", "Med-Slow", "Med-Fast", "Fast"]]
        gcp_gaps = [r["conditional"]["group_cp"][g]["coverage"] < 0.89
                    for g in ["Slow", "Med-Slow", "Med-Fast", "Fast"]]

        print(f"\n{s.upper()}:")
        print(f"  Top-1: {r['top1']:.1%} | Std CP: {std['coverage']:.1%} (size {std['avg_set_size']:.2f})")
        print(f"  Std CP gaps: {sum(std_gaps)}/4 groups under 89%")
        print(f"  GCP gaps: {sum(gcp_gaps)}/4 groups under 89% (size {gcp['avg_set_size']:.2f})")


if __name__ == "__main__":
    main()
