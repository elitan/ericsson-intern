"""
Paper experiments v5: Ensemble baseline + latency measurements.

Changes from v4:
- Add deep ensemble baseline (5 models, averaged softmax)
- Add wall-clock latency measurements for CP operations
"""

import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
import json
from typing import List, Dict
import torch

from src.handover.synthetic_data import generate_dataset, NetworkConfig, MobilityConfig
from src.handover.predictor import train_predictor, get_softmax_scores, prepare_features, HandoverMLP
from src.handover.conformal import calibrate_threshold, predict_sets, evaluate_cp


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


def train_predictor_silent(
    data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> HandoverMLP:
    """Train predictor without printing (for ensemble)."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    rsrp, serving, optimal, speed = prepare_features(data)
    n_cells = data["n_cells"]

    train_ds = TensorDataset(
        rsrp[train_idx], serving[train_idx], optimal[train_idx], speed[train_idx]
    )
    val_ds = TensorDataset(
        rsrp[val_idx], serving[val_idx], optimal[val_idx], speed[val_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = HandoverMLP(n_cells)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for rsrp_b, serving_b, optimal_b, speed_b in train_loader:
            optimizer.zero_grad()
            logits = model(rsrp_b, serving_b, speed_b)
            loss = criterion(logits, optimal_b)
            loss.backward()
            optimizer.step()

    return model


def train_ensemble(
    data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_models: int = 5,
    n_epochs: int = 20,
) -> List[HandoverMLP]:
    """Train ensemble of models with different seeds."""
    models = []
    ensemble_seeds = [100, 200, 300, 400, 500]
    for i in range(n_models):
        model = train_predictor_silent(
            data, train_idx, val_idx,
            n_epochs=n_epochs, seed=ensemble_seeds[i]
        )
        models.append(model)
    return models


def get_ensemble_probs(
    models: List[HandoverMLP],
    data: dict,
    idx: np.ndarray,
) -> np.ndarray:
    """Get averaged softmax from ensemble."""
    all_probs = []
    for model in models:
        probs = get_softmax_scores(model, data, idx)
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


def calibrate_ensemble_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_coverage: float = 0.90,
) -> float:
    """Calibrate threshold for ensemble to achieve target coverage.

    Uses cumulative probability approach: include classes in descending
    probability order until cumulative prob >= threshold.
    """
    n = len(labels)

    thresholds = np.linspace(0.5, 0.99, 50)

    for thresh in thresholds:
        covered = 0
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0
            pred_set = set()
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i, idx]
                if cumsum >= thresh:
                    break
            if labels[i] in pred_set:
                covered += 1

        coverage = covered / n
        if coverage >= target_coverage:
            return thresh

    return 0.5


def predict_ensemble_sets(
    probs: np.ndarray,
    threshold: float,
) -> List[set]:
    """Create prediction sets from ensemble using calibrated threshold."""
    pred_sets = []
    for i in range(len(probs)):
        sorted_idx = np.argsort(probs[i])[::-1]
        cumsum = 0
        pred_set = set()
        for idx in sorted_idx:
            pred_set.add(idx)
            cumsum += probs[i, idx]
            if cumsum >= threshold:
                break
        pred_sets.append(pred_set)
    return pred_sets


def measure_latency(
    model: HandoverMLP,
    data: dict,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    cal_labels: np.ndarray,
    n_trials: int = 100,
) -> Dict:
    """Measure wall-clock latency for CP operations."""

    cal_probs = get_softmax_scores(model, data, cal_idx)
    test_probs = get_softmax_scores(model, data, test_idx)

    calibration_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        threshold = calibrate_threshold(cal_probs, cal_labels, alpha=0.10)
        calibration_times.append(time.perf_counter() - start)

    threshold = calibrate_threshold(cal_probs, cal_labels, alpha=0.10)

    prediction_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        pred_sets = predict_sets(test_probs, threshold)
        prediction_times.append(time.perf_counter() - start)

    n_test = len(test_idx)
    per_sample_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for i in range(min(1000, n_test)):
            pred_set = {j for j in range(test_probs.shape[1])
                       if test_probs[i, j] >= 1 - threshold}
        per_sample_times.append((time.perf_counter() - start) / min(1000, n_test))

    rsrp, serving, optimal, speed = prepare_features(data)
    model.eval()
    inference_times = []
    batch_size = 1000
    with torch.no_grad():
        for _ in range(n_trials):
            start = time.perf_counter()
            logits = model(
                rsrp[test_idx[:batch_size]],
                serving[test_idx[:batch_size]],
                speed[test_idx[:batch_size]]
            )
            probs = torch.softmax(logits, dim=1)
            inference_times.append((time.perf_counter() - start) / batch_size)

    return {
        "calibration_ms": np.mean(calibration_times) * 1000,
        "calibration_std_ms": np.std(calibration_times) * 1000,
        "batch_prediction_ms": np.mean(prediction_times) * 1000,
        "batch_prediction_std_ms": np.std(prediction_times) * 1000,
        "per_sample_prediction_us": np.mean(per_sample_times) * 1e6,
        "per_sample_prediction_std_us": np.std(per_sample_times) * 1e6,
        "nn_inference_per_sample_us": np.mean(inference_times) * 1e6,
        "nn_inference_std_us": np.std(inference_times) * 1e6,
        "n_cal_samples": len(cal_idx),
        "n_test_samples": len(test_idx),
    }


def run_single_seed(scenario_key: str, seed: int, n_traj: int = 600) -> Dict:
    """Run experiment for one seed with ensemble + latency."""
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

    cal_labels = data["optimal_cell"][cal_idx]
    test_labels = data["optimal_cell"][test_idx]

    print("    Training single model...", end=" ", flush=True)
    model = train_predictor_silent(data, train_idx, cal_idx, n_epochs=20, seed=seed)

    cal_probs = get_softmax_scores(model, data, cal_idx)
    test_probs = get_softmax_scores(model, data, test_idx)

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    print(f"Top-1: {top1_acc:.3f}", end=" ", flush=True)

    threshold_cp = calibrate_threshold(cal_probs, cal_labels, alpha=0.10)
    pred_sets_cp = predict_sets(test_probs, threshold_cp)
    cp_eval = evaluate_cp(pred_sets_cp, test_labels)

    results = {
        "n_cells": n_cells,
        "top1_acc": float(top1_acc),
        "cp_coverage": cp_eval["coverage"],
        "cp_size": cp_eval["avg_set_size"],
    }

    print("Ensemble...", end=" ", flush=True)
    ensemble_models = train_ensemble(data, train_idx, cal_idx, n_models=5, n_epochs=20)

    cal_probs_ens = get_ensemble_probs(ensemble_models, data, cal_idx)
    test_probs_ens = get_ensemble_probs(ensemble_models, data, test_idx)

    ens_top1_acc = (test_probs_ens.argmax(axis=1) == test_labels).mean()

    threshold_ens = calibrate_ensemble_threshold(cal_probs_ens, cal_labels, target_coverage=0.90)
    pred_sets_ens = predict_ensemble_sets(test_probs_ens, threshold_ens)
    ens_eval = evaluate_cp(pred_sets_ens, test_labels)

    results["ens_top1_acc"] = float(ens_top1_acc)
    results["ens_coverage"] = ens_eval["coverage"]
    results["ens_size"] = ens_eval["avg_set_size"]
    results["ens_threshold"] = float(threshold_ens)

    print(f"Ens: {ens_eval['coverage']:.3f}", end=" ", flush=True)

    print("Latency...", end=" ", flush=True)
    latency = measure_latency(model, data, cal_idx, test_idx, cal_labels, n_trials=50)
    results.update({f"latency_{k}": v for k, v in latency.items()})

    print("Done")

    return results


def aggregate_results(seed_results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    keys = seed_results[0].keys()
    agg = {}
    for key in keys:
        if key in ["n_cells"]:
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
        print(f"  Seed {seed} ({i+1}/{len(seeds)}):", end=" ", flush=True)
        result = run_single_seed(scenario_key, seed, n_traj)
        seed_results.append(result)

    agg = aggregate_results(seed_results)
    agg["scenario"] = cfg.name

    print(f"\n  Summary:")
    print(f"    Single Model CP: {agg['cp_coverage_mean']:.3f} ± {agg['cp_coverage_std']:.3f} (size {agg['cp_size_mean']:.2f})")
    print(f"    Ensemble (5 models): {agg['ens_coverage_mean']:.3f} ± {agg['ens_coverage_std']:.3f} (size {agg['ens_size_mean']:.2f})")
    print(f"    Latency - Calibration: {agg['latency_calibration_ms_mean']:.2f}ms")
    print(f"    Latency - Per-sample CP: {agg['latency_per_sample_prediction_us_mean']:.2f}μs")
    print(f"    Latency - NN inference: {agg['latency_nn_inference_per_sample_us_mean']:.2f}μs")

    return agg


def generate_latex_output(all_results: Dict):
    """Generate LaTeX table for ensemble comparison."""

    print("\n" + "="*70)
    print("TABLE: CP vs Ensemble Comparison")
    print("="*70)

    print(r"""
\begin{table}[t]
\caption{Standard CP vs Deep Ensemble ($\alpha=0.1$, 5 seeds)}
\label{tab:ensemble}
\centering
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Single Model + CP} & \multicolumn{2}{c}{Ensemble (5 models)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Scenario & Coverage & Size & Coverage & Size \\
\midrule""")

    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"{s.capitalize()} & "
              f"${r['cp_coverage_mean']:.2f} \\pm {r['cp_coverage_std']:.2f}$ & "
              f"${r['cp_size_mean']:.1f} \\pm {r['cp_size_std']:.1f}$ & "
              f"${r['ens_coverage_mean']:.2f} \\pm {r['ens_coverage_std']:.2f}$ & "
              f"${r['ens_size_mean']:.1f} \\pm {r['ens_size_std']:.1f}$ \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*70)
    print("Latency Numbers (for Discussion section)")
    print("="*70)

    r = all_results["medium"]
    print(f"""
Computational overhead measurements (Medium scenario, {r['latency_n_cal_samples_mean']:.0f} calibration samples):
- CP calibration (one-time): {r['latency_calibration_ms_mean']:.2f} ± {r['latency_calibration_std_ms_mean']:.2f} ms
- CP per-sample prediction: {r['latency_per_sample_prediction_us_mean']:.2f} ± {r['latency_per_sample_prediction_std_us_mean']:.2f} μs
- NN inference per-sample: {r['latency_nn_inference_per_sample_us_mean']:.2f} ± {r['latency_nn_inference_std_us_mean']:.2f} μs
- CP overhead vs NN: {r['latency_per_sample_prediction_us_mean'] / r['latency_nn_inference_per_sample_us_mean'] * 100:.1f}%
""")


def main():
    print("="*70)
    print("  PAPER EXPERIMENTS v5: Ensemble Baseline + Latency")
    print("="*70)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    seeds = [42, 123, 456, 789, 1011]

    all_results = {}
    for scenario in ["easy", "medium", "hard"]:
        all_results[scenario] = run_scenario_with_seeds(scenario, seeds, n_traj=600)

    with open(figures_dir / "paper_results_v5.json", "w") as f:
        json.dump({s: {k: v for k, v in r.items()} for s, r in all_results.items()}, f, indent=2)
    print(f"\nSaved: figures/paper_results_v5.json")

    generate_latex_output(all_results)

    print("\n" + "="*70)
    print("  KEY FINDINGS")
    print("="*70)

    print("\n1. CP vs Ensemble Coverage:")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"   {s}: CP={r['cp_coverage_mean']:.1%}±{r['cp_coverage_std']:.1%}, "
              f"Ensemble={r['ens_coverage_mean']:.1%}±{r['ens_coverage_std']:.1%}")

    print("\n2. CP vs Ensemble Set Size:")
    for s in ["easy", "medium", "hard"]:
        r = all_results[s]
        print(f"   {s}: CP={r['cp_size_mean']:.1f}, Ensemble={r['ens_size_mean']:.1f}")

    print("\n3. Training Cost:")
    print("   Single model + CP: 1x training")
    print("   Ensemble: 5x training")

    print("\n4. Formal Guarantees:")
    print("   CP: ✓ Distribution-free coverage guarantee")
    print("   Ensemble: ✗ No formal guarantee (empirical only)")


if __name__ == "__main__":
    main()
