"""
End-to-end experiment: Conformal Prediction for Handover.

1. Generate synthetic handover data
2. Train handover predictor
3. Calibrate conformal prediction
4. Evaluate coverage, set size, etc.
5. Compare standard CP vs ACI (adaptive)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.handover.synthetic_data import generate_dataset, NetworkConfig, MobilityConfig
from src.handover.predictor import train_predictor, get_softmax_scores, prepare_features
from src.handover.conformal import (
    calibrate_threshold,
    predict_sets,
    evaluate_cp,
    run_aci_online,
)


def main():
    print("=" * 60)
    print("Conformal Prediction for Handover - Experiment")
    print("=" * 60)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    print("\n[1] Generating synthetic handover data...")
    network_config = NetworkConfig(
        n_gnb_x=4,
        n_gnb_y=4,
        cell_radius=150.0,
        noise_std_db=6.0,
    )
    mobility_config = MobilityConfig(n_trajectories=1000, trajectory_length=100)
    prediction_horizon = 10
    measurement_noise = 4.0
    data = generate_dataset(
        network_config, mobility_config, seed=42,
        prediction_horizon=prediction_horizon,
        measurement_noise_db=measurement_noise,
    )
    print(f"    Prediction horizon: {prediction_horizon} steps")
    print(f"    Measurement noise: {measurement_noise} dB")

    n_samples = len(data["rsrp"])
    n_cells = data["n_cells"]
    print(f"    Samples: {n_samples}")
    print(f"    Cells: {n_cells}")
    print(f"    Trajectories: {mobility_config.n_trajectories}")

    n_handovers = np.sum(np.diff(data["serving_cell"]) != 0)
    print(f"    Total handovers: {n_handovers}")

    print("\n[2] Splitting data (train/cal/test by trajectory)...")
    n_traj = mobility_config.n_trajectories
    traj_ids = data["trajectory_id"]

    train_traj = set(range(int(0.6 * n_traj)))
    cal_traj = set(range(int(0.6 * n_traj), int(0.8 * n_traj)))
    test_traj = set(range(int(0.8 * n_traj), n_traj))

    train_idx = np.array([i for i, t in enumerate(traj_ids) if t in train_traj])
    cal_idx = np.array([i for i, t in enumerate(traj_ids) if t in cal_traj])
    test_idx = np.array([i for i, t in enumerate(traj_ids) if t in test_traj])

    print(f"    Train: {len(train_idx)}, Cal: {len(cal_idx)}, Test: {len(test_idx)}")

    print("\n[3] Training handover predictor...")
    model = train_predictor(data, train_idx, cal_idx, n_epochs=15, batch_size=512)

    print("\n[4] Getting softmax scores...")
    cal_probs = get_softmax_scores(model, data, cal_idx)
    test_probs = get_softmax_scores(model, data, test_idx)

    cal_labels = data["optimal_cell"][cal_idx]
    test_labels = data["optimal_cell"][test_idx]

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    print(f"    Top-1 accuracy: {top1_acc:.4f}")

    print("\n[5] Standard Split Conformal Prediction...")
    alpha = 0.1
    threshold = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    print(f"    Threshold (alpha={alpha}): {threshold:.4f}")

    pred_sets = predict_sets(test_probs, threshold)
    results = evaluate_cp(pred_sets, test_labels)

    print(f"    Coverage: {results['coverage']:.4f} (target: {1-alpha:.2f})")
    print(f"    Avg set size: {results['avg_set_size']:.3f}")
    print(f"    Size=1 fraction: {results['size_1_frac']:.3f}")

    print("\n[6] Adaptive Conformal Inference (ACI)...")
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
        test_probs_seq, test_labels_seq, cal_scores, alpha=alpha, gamma=0.01
    )

    print(f"    ACI Coverage: {aci_results['coverage']:.4f}")
    print(f"    ACI Avg set size: {aci_results['avg_set_size']:.3f}")

    print("\n[7] Coverage by UE speed...")
    test_speeds = data["ue_speed"][test_idx]
    speed_quartiles = np.percentile(test_speeds, [25, 50, 75])

    def get_speed_bin(s):
        if s < speed_quartiles[0]:
            return 0
        elif s < speed_quartiles[1]:
            return 1
        elif s < speed_quartiles[2]:
            return 2
        else:
            return 3

    speed_bins = np.array([get_speed_bin(s) for s in test_speeds])
    bin_names = ["Slow", "Medium-Slow", "Medium-Fast", "Fast"]

    print("    Speed bin coverage:")
    for b in range(4):
        mask = speed_bins == b
        cov = np.mean([test_labels[i] in pred_sets[i] for i in np.where(mask)[0]])
        sizes = [len(pred_sets[i]) for i in np.where(mask)[0]]
        print(f"      {bin_names[b]}: coverage={cov:.3f}, avg_size={np.mean(sizes):.2f}")

    print("\n[8] Generating figures...")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    sizes = [len(s) for s in pred_sets]
    axes[0].hist(sizes, bins=range(1, max(sizes) + 2), edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Prediction Set Size")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Set Sizes (Standard CP)")
    axes[0].axvline(np.mean(sizes), color="red", linestyle="--", label=f"Mean={np.mean(sizes):.2f}")
    axes[0].legend()

    coverages_by_bin = []
    for b in range(4):
        mask = speed_bins == b
        cov = np.mean([test_labels[i] in pred_sets[i] for i in np.where(mask)[0]])
        coverages_by_bin.append(cov)
    axes[1].bar(bin_names, coverages_by_bin, color=["green" if c >= 0.9 else "red" for c in coverages_by_bin])
    axes[1].axhline(1 - alpha, color="black", linestyle="--", label=f"Target={1-alpha}")
    axes[1].set_ylabel("Coverage")
    axes[1].set_title("Conditional Coverage by Speed")
    axes[1].set_ylim(0.7, 1.0)
    axes[1].legend()

    window = 500
    rolling_cov = np.convolve(aci_results["coverages"], np.ones(window) / window, mode="valid")
    axes[2].plot(rolling_cov, alpha=0.7)
    axes[2].axhline(1 - alpha, color="red", linestyle="--", label=f"Target={1-alpha}")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Rolling Coverage")
    axes[2].set_title(f"ACI Rolling Coverage (window={window})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "experiment_results.png", dpi=150)
    print(f"    Saved: {figures_dir / 'experiment_results.png'}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Top-1 Accuracy:     {top1_acc:.4f}")
    print(f"Standard CP:        coverage={results['coverage']:.4f}, set_size={results['avg_set_size']:.3f}")
    print(f"ACI (adaptive):     coverage={aci_results['coverage']:.4f}, set_size={aci_results['avg_set_size']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
