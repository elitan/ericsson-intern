"""Visualization for temporal beam prediction results.

Plots: CP set size vs time, coverage vs speed, L1-RSRP CDF,
blockage demo, accuracy vs speed, top-k bars.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beampred.config import FIGURES_DIR, CONFORMAL_ALPHA


TEMPORAL_FIG_DIR = os.path.join(FIGURES_DIR, "temporal")


def _savefig(fig, name):
    os.makedirs(TEMPORAL_FIG_DIR, exist_ok=True)
    path = os.path.join(TEMPORAL_FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cp_vs_time(timesteps, set_sizes, blockage_states=None, title="CP Set Size Over Time"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(timesteps, set_sizes, linewidth=0.8, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CP set size")
    ax.set_title(title)

    if blockage_states is not None:
        blocked = blockage_states.astype(bool)
        if np.any(blocked):
            ymin, ymax = ax.get_ylim()
            ax.fill_between(timesteps, ymin, ymax, where=blocked,
                            alpha=0.2, color="red", label="Blockage")
            ax.legend(loc="upper right")

    ax.grid(True, alpha=0.3)
    _savefig(fig, "cp_set_size_vs_time.pdf")


def plot_coverage_vs_speed(all_outputs):
    speeds = sorted(all_outputs.keys())
    coverages = [all_outputs[s]["coverage"] for s in speeds]
    mean_sizes = [all_outputs[s]["sizes"].mean() for s in speeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar([str(s) for s in speeds], coverages, color="tab:blue", alpha=0.8)
    ax1.axhline(y=1 - CONFORMAL_ALPHA, color="red", linestyle="--",
                label=f"Target ({1-CONFORMAL_ALPHA:.0%})")
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Coverage")
    ax1.set_title("CP Coverage vs Speed")
    ax1.legend()
    ax1.set_ylim(0.8, 1.0)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar([str(s) for s in speeds], mean_sizes, color="tab:orange", alpha=0.8)
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Mean set size")
    ax2.set_title("CP Set Size vs Speed")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _savefig(fig, "coverage_vs_speed.pdf")


def plot_l1_rsrp_cdf(gap_db, speed_kmh=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_gap = np.sort(gap_db)
    cdf = np.arange(1, len(sorted_gap) + 1) / len(sorted_gap)

    label = f"{speed_kmh} km/h" if speed_kmh else "L1-RSRP Gap"
    ax.plot(sorted_gap, cdf, linewidth=1.5, label=label)

    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="1 dB target")
    ax.set_xlabel("L1-RSRP Gap (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("L1-RSRP Gap Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    suffix = f"_{speed_kmh}kmh" if speed_kmh else ""
    _savefig(fig, f"l1_rsrp_cdf{suffix}.pdf")


def plot_blockage_demo(demo_result):
    timesteps = demo_result["timesteps"]
    set_sizes = demo_result["set_sizes"]
    pred_rsrp = demo_result["pred_rsrp"]
    oracle_rsrp = demo_result["oracle_rsrp"]
    blockage_states = demo_result["blockage_states"]
    onset = demo_result["blockage_onset_time"]
    lead_time = demo_result["lead_time_s"]
    threshold = demo_result["expansion_threshold"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    if "set_sizes_raw" in demo_result:
        ax.plot(timesteps, demo_result["set_sizes_raw"], linewidth=0.4,
                color="tab:blue", alpha=0.3, label="Raw")
    ax.plot(timesteps, set_sizes, linewidth=0.8, color="tab:blue",
            label="Smoothed (5-pt MA)")
    ax.axhline(y=threshold, color="orange", linestyle="--", alpha=0.7,
               label=f"Expansion threshold ({threshold:.0f})")
    ax.axvline(x=onset, color="red", linestyle="-", alpha=0.7, label="Blockage onset")
    if lead_time > 0:
        ax.axvline(x=onset - lead_time, color="green", linestyle="--",
                   alpha=0.7, label=f"Early warning ({lead_time*1000:.0f}ms)")
    ax.set_ylabel("CP set size")
    ax.set_title(f"Blockage Early Warning Demo ({demo_result['speed_kmh']} km/h)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(timesteps, pred_rsrp, linewidth=0.8, label="Predicted beam", color="tab:blue")
    ax.plot(timesteps, oracle_rsrp, linewidth=0.8, label="Oracle beam",
            color="tab:green", alpha=0.7)
    ax.axvline(x=onset, color="red", linestyle="-", alpha=0.7)
    ax.set_ylabel("RSRP (dB)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    blocked = blockage_states.astype(bool)
    ax.fill_between(timesteps, 0, 1, where=blocked, alpha=0.5, color="red", label="Blocked")
    ax.fill_between(timesteps, 0, 1, where=~blocked, alpha=0.3, color="green", label="Clear")
    ax.set_ylabel("Blockage state")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Clear", "Blocked"])
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _savefig(fig, "blockage_demo.pdf")


def plot_accuracy_vs_speed(all_outputs):
    """Top-1 accuracy vs speed for all methods."""
    speeds = sorted(all_outputs.keys())

    methods = {
        "Last-beam": [all_outputs[s]["lastbeam_results"]["top1"] for s in speeds],
        "MLP": [all_outputs[s]["mlp_results"]["top1"] for s in speeds],
        "LSTM": [all_outputs[s]["lstm_results"]["top1"] for s in speeds],
        "Transformer": [all_outputs[s]["tf_results"]["top1"] for s in speeds],
    }

    colors = {"Last-beam": "tab:gray", "MLP": "tab:orange",
              "LSTM": "tab:blue", "Transformer": "tab:green"}
    markers = {"Last-beam": "s", "MLP": "^", "LSTM": "o", "Transformer": "D"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, accs in methods.items():
        ax.plot(speeds, accs, marker=markers[name], label=name,
                color=colors[name], linewidth=2, markersize=8)

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Beam Prediction Accuracy vs UE Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(speeds)

    fig.tight_layout()
    _savefig(fig, "accuracy_vs_speed.pdf")


def plot_topk_bars(all_outputs):
    """Top-1/3/5 grouped bars for each speed (best model)."""
    speeds = sorted(all_outputs.keys())
    top1 = [all_outputs[s]["best_results"]["top1"] for s in speeds]
    top3 = [all_outputs[s]["best_results"]["top3"] for s in speeds]
    top5 = [all_outputs[s]["best_results"]["top5"] for s in speeds]

    x = np.arange(len(speeds))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, top1, w, label="Top-1", color="tab:blue", alpha=0.8)
    ax.bar(x, top3, w, label="Top-3", color="tab:orange", alpha=0.8)
    ax.bar(x + w, top5, w, label="Top-5", color="tab:green", alpha=0.8)

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k Accuracy (Best Model per Speed)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in speeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _savefig(fig, "topk_accuracy.pdf")


def plot_speed_comparison_table(all_outputs, oh_results=None):
    speeds = sorted(all_outputs.keys())

    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(speeds)))
    ax.axis("off")

    headers = ["Speed\n(km/h)", "Last-beam\nTop-1", "MLP\nTop-1", "LSTM\nTop-1",
               "TF\nTop-1", "Best", "CP\nCoverage", "Mean\nSet Size"]
    if oh_results:
        headers.append("Overhead\nReduction")

    rows = []
    for s in speeds:
        out = all_outputs[s]
        row = [
            f"{s}",
            f"{out['lastbeam_results']['top1']:.4f}",
            f"{out['mlp_results']['top1']:.4f}",
            f"{out['lstm_results']['top1']:.4f}",
            f"{out['tf_results']['top1']:.4f}",
            out["best_name"],
            f"{out['coverage']:.4f}",
            f"{out['sizes'].mean():.2f}",
        ]
        if oh_results and s in oh_results:
            row.append(f"{oh_results[s]['mean_reduction']*100:.1f}%")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    _savefig(fig, "speed_comparison_table.pdf")
