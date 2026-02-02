import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beampred.config import SNR_VALUES_DB, FIGURES_DIR, N_NARROW_BEAMS, N_WIDE_BEAMS
from beampred.codebook import get_narrow_codebook, get_wide_codebook, beam_angles
from beampred.utils import array_response_vector


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_beam_patterns():
    setup_style()
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 5))

    theta = np.linspace(-np.pi / 2, np.pi / 2, 500)
    narrow_cb = get_narrow_codebook()
    wide_cb = get_wide_codebook()

    for i in range(N_NARROW_BEAMS):
        gains = np.array([np.abs(narrow_cb[i] @ array_response_vector(t)) ** 2 for t in theta])
        gains_db = 10 * np.log10(np.maximum(gains, 1e-10))
        gains_db = np.maximum(gains_db, -30)
        axes[0].plot(theta + np.pi / 2, gains_db + 30, alpha=0.4, linewidth=0.5)
    axes[0].set_title(f"Narrow codebook ({N_NARROW_BEAMS} beams)")
    axes[0].set_thetamin(0)
    axes[0].set_thetamax(180)

    for i in range(N_WIDE_BEAMS):
        gains = np.array([np.abs(wide_cb[i] @ array_response_vector(t)) ** 2 for t in theta])
        gains_db = 10 * np.log10(np.maximum(gains, 1e-10))
        gains_db = np.maximum(gains_db, -30)
        axes[1].plot(theta + np.pi / 2, gains_db + 30, linewidth=1.5)
    axes[1].set_title(f"Wide codebook ({N_WIDE_BEAMS} beams)")
    axes[1].set_thetamin(0)
    axes[1].set_thetamax(180)

    fig.suptitle("DFT Codebook Beam Patterns", fontsize=14)
    save_fig(fig, "beam-patterns.png")


def plot_topk_vs_snr(all_results, std_results=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [r.get("method", f"Method {i}") for i, r in enumerate(all_results)]
    top1_vals = [r["top1"] for r in all_results]
    top3_vals = [r["top3"] for r in all_results]
    top5_vals = [r["top5"] for r in all_results]

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, top1_vals, width, label="Top-1", color="C0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, top3_vals, width, label="Top-3", color="C1", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, top5_vals, width, label="Top-5", color="C2", edgecolor="black", linewidth=0.5)

    if std_results is not None:
        top1_err = [std_results[i].get("top1", 0) for i in range(len(all_results))]
        top3_err = [std_results[i].get("top3", 0) for i in range(len(all_results))]
        top5_err = [std_results[i].get("top5", 0) for i in range(len(all_results))]
        ax.errorbar(x - width, top1_vals, yerr=top1_err, fmt="none", color="black", capsize=3)
        ax.errorbar(x, top3_vals, yerr=top3_err, fmt="none", color="black", capsize=3)
        ax.errorbar(x + width, top5_vals, yerr=top5_err, fmt="none", color="black", capsize=3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k Beam Prediction Accuracy")
    ax.legend()
    ax.set_ylim(0, 1.15)
    save_fig(fig, "topk-accuracy.png")


def plot_spectral_efficiency(all_results, std_results=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    snrs = SNR_VALUES_DB
    markers = ["o", "s", "^", "D", "v", "P"]
    styles = ["-", "--", "-.", ":", "-", "--"]

    se_exh = [all_results[0]["snr_results"][s]["se_exhaustive"] for s in snrs]
    ax.plot(snrs, se_exh, "ko-", label="Exhaustive (optimal)", linewidth=2)

    for idx, results in enumerate(all_results):
        name = results.get("method", f"Method {idx}")
        se = [results["snr_results"][s]["se_method"] for s in snrs]
        ax.plot(snrs, se, f"{markers[idx%6]}{styles[idx%6]}",
                label=name, linewidth=1.5)
        if std_results is not None and idx < len(std_results):
            se_std = [std_results[idx].get("snr_results", {}).get(s, {}).get("se_method", 0) for s in snrs]
            se_arr = np.array(se)
            se_std_arr = np.array(se_std)
            ax.fill_between(snrs, se_arr - se_std_arr, se_arr + se_std_arr, alpha=0.15)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Spectral efficiency (bps/Hz)")
    ax.set_title("Spectral Efficiency Comparison")
    ax.legend(fontsize=8)
    ax.set_xticks(snrs)
    save_fig(fig, "spectral-efficiency.png")


def plot_throughput(all_results, std_results=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    snrs = SNR_VALUES_DB
    markers = ["o", "s", "^", "D", "v", "P"]
    styles = ["-", "--", "-.", ":", "-", "--"]

    tp_exh = [all_results[0]["snr_results"][s]["tp_exhaustive"] for s in snrs]
    ax.plot(snrs, tp_exh, "ko-", label="Exhaustive (64 slots)", linewidth=2)

    for idx, results in enumerate(all_results):
        name = results.get("method", f"Method {idx}")
        oh = results.get("overhead", "?")
        tp = [results["snr_results"][s]["tp_method"] for s in snrs]
        ax.plot(snrs, tp, f"{markers[idx%6]}{styles[idx%6]}",
                label=f"{name} ({oh} slots)", linewidth=1.5)
        if std_results is not None and idx < len(std_results):
            tp_std = [std_results[idx].get("snr_results", {}).get(s, {}).get("tp_method", 0) for s in snrs]
            tp_arr = np.array(tp)
            tp_std_arr = np.array(tp_std)
            ax.fill_between(snrs, tp_arr - tp_std_arr, tp_arr + tp_std_arr, alpha=0.15)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Effective throughput (bps/Hz)")
    ax.set_title("Effective Throughput (including beam search overhead)")
    ax.legend(fontsize=8)
    ax.set_xticks(snrs)
    save_fig(fig, "throughput.png")


def plot_confusion_matrix(results):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    cm = results["confusion"]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Prediction rate")
    ax.set_xlabel("Predicted beam")
    ax.set_ylabel("True beam")
    ax.set_title("Beam Prediction Confusion Matrix (MLP)")
    save_fig(fig, "confusion-matrix.png")


def plot_accuracy_vs_distance(all_results, std_results=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ref = None
    for r in all_results:
        if r["distance_results"]:
            ref = r
            break
    if ref is None:
        plt.close(fig)
        return

    bins = sorted(ref["distance_results"].keys())
    labels = [f"{ref['distance_results'][b]['range'][0]:.0f}-{ref['distance_results'][b]['range'][1]:.0f}m" for b in bins]

    for idx, results in enumerate(all_results):
        name = results.get("method", f"Method {idx}")
        dist_res = results["distance_results"]
        if not dist_res:
            continue
        top1_vals = [dist_res[b]["top1"] if b in dist_res else 0 for b in bins]

        x = np.arange(len(bins))
        w = 0.8 / len(all_results)
        offset = (idx - len(all_results) / 2 + 0.5) * w
        bar_container = ax.bar(x + offset, top1_vals, w, label=name)

        if std_results is not None and idx < len(std_results):
            std_dist = std_results[idx].get("distance_results", {})
            yerr = [std_dist.get(b, {}).get("top1", 0) for b in bins]
            ax.errorbar(x + offset, top1_vals, yerr=yerr, fmt="none", color="black", capsize=2)

    x = np.arange(len(bins))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distance range")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Beam Prediction Accuracy vs Distance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    save_fig(fig, "accuracy-vs-distance.png")


def plot_error_distance_histogram(error_results):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    hist = error_results["error_histogram"]
    max_show = min(20, len(hist))
    x = np.arange(1, max_show + 1)
    ax.bar(x, hist[:max_show], color="C3", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Beam error distance |predicted - true|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Beam Prediction Error Magnitudes")
    ax.set_xticks(x)
    save_fig(fig, "error-distance-histogram.png")


def plot_cost_vs_error_distance(error_results):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    loss = error_results["gain_loss_per_distance"]
    counts = error_results["gain_loss_counts"]
    max_show = min(20, len(loss))
    x = np.arange(1, max_show + 1)
    mask = (counts[:max_show] > 0) & (counts[:max_show] >= 30)
    bars = ax.bar(x[mask], loss[:max_show][mask], color="C1", edgecolor="black", linewidth=0.5)

    for bar, cnt in zip(bars, counts[:max_show][mask]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"n={int(cnt)}", ha="center", va="bottom", fontsize=7)

    if mask.sum() >= 3:
        from numpy.polynomial import polynomial as P
        x_fit = x[mask].astype(float)
        y_fit = loss[:max_show][mask]
        coeffs = P.polyfit(x_fit, y_fit, 2)
        x_smooth = np.linspace(x_fit.min(), x_fit.max(), 50)
        y_smooth = P.polyval(x_smooth, coeffs)
        ax.plot(x_smooth, y_smooth, "r--", linewidth=1.5, alpha=0.7, label="Trend")
        ax.legend()

    ax.set_xlabel("Beam error distance")
    ax.set_ylabel("Average gain loss (dB)")
    ax.set_title("Gain Loss vs Beam Error Distance")
    ax.set_xticks(x)
    save_fig(fig, "cost-vs-error-distance.png")


def plot_adaptive_tradeoff(sweep_results):
    setup_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    thresholds = [r["threshold"] for r in sweep_results]
    accuracies = [r["accuracy"] for r in sweep_results]
    overheads = [r["avg_overhead"] for r in sweep_results]

    ax1.plot(thresholds, accuracies, "o-", color="C0", linewidth=2, label="Accuracy")
    ax1.set_xlabel("Confidence threshold (max prediction set size)")
    ax1.set_ylabel("Accuracy", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0.5, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, overheads, "s--", color="C3", linewidth=2, label="Avg overhead")
    ax2.set_ylabel("Average overhead (slots)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Adaptive Fallback: Accuracy vs Overhead Trade-off")
    ax1.set_xticks(thresholds)
    save_fig(fig, "adaptive-tradeoff.png")


def plot_prediction_set_sizes(sizes, sizes_beam_aware=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    max_size = min(int(sizes.max()), 30)
    bins = np.arange(1, max_size + 2) - 0.5
    ax.hist(sizes, bins=bins, color="C2", edgecolor="black", linewidth=0.5, density=True,
            alpha=0.7, label=f"Standard (mean={np.mean(sizes):.1f})")

    if sizes_beam_aware is not None:
        ax.hist(sizes_beam_aware, bins=bins, color="C4", edgecolor="black", linewidth=0.5,
                density=True, alpha=0.5, label=f"Beam-aware (mean={np.mean(sizes_beam_aware):.1f})")

    ax.axvline(np.mean(sizes), color="red", linestyle="--", linewidth=2,
               label=f"Std mean: {np.mean(sizes):.1f}")
    ax.set_xlabel("Prediction set size")
    ax.set_ylabel("Density")
    ax.set_title("Conformal Prediction Set Size Distribution")
    ax.legend()
    save_fig(fig, "prediction-set-sizes.png")


def generate_all_figures(all_results, mlp_results=None, error_results=None,
                         sweep_results=None, set_sizes_arr=None,
                         std_results=None, set_sizes_beam_aware=None):
    print("Generating figures...")
    plot_beam_patterns()
    plot_topk_vs_snr(all_results, std_results=std_results)
    plot_spectral_efficiency(all_results, std_results=std_results)
    plot_throughput(all_results, std_results=std_results)
    if mlp_results is not None:
        plot_confusion_matrix(mlp_results)
    plot_accuracy_vs_distance(all_results, std_results=std_results)
    if error_results is not None:
        plot_error_distance_histogram(error_results)
        plot_cost_vs_error_distance(error_results)
    if sweep_results is not None:
        plot_adaptive_tradeoff(sweep_results)
    if set_sizes_arr is not None:
        plot_prediction_set_sizes(set_sizes_arr, sizes_beam_aware=set_sizes_beam_aware)
    print("All figures saved.")
