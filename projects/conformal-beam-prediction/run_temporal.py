"""Temporal beam prediction pipeline with conformal guarantees.

Full pipeline per speed:
1. Generate/load Sionna channels
2. Build temporal sequences
3. Evaluate last-beam baseline
4. Train MLP, LSTM, Transformer
5. Evaluate top-1/3/5 accuracy
6. Per-speed conformal calibration (primary) + cross-speed (secondary)
7. Compute 3GPP KPIs
8. Run blockage demo
9. Generate figures
"""
import argparse
import time
import numpy as np
import torch

from beampred import config
from beampred.config import (
    SPEEDS_KMH, SEQ_LEN, N_NARROW_BEAMS, N_WIDE_BEAMS,
    CONFORMAL_ALPHA, FIGURES_DIR
)
from beampred.sionna_channel import generate_sionna_channels, save_channels, load_channels
from beampred.temporal_dataset import get_temporal_dataloaders
from beampred.temporal_predictor import (
    BeamMLP, BeamLSTM, BeamTemporalTransformer, train_temporal_model
)
from beampred.conformal import calibrate, predict_sets, set_sizes, coverage
from beampred.kpi import compute_all_kpis, format_kpi_table, overhead_slots
from beampred.codebook import generate_dft_codebook
from beampred.blockage import apply_blockage_to_channels, run_blockage_demo
from beampred.temporal_visualize import (
    plot_cp_vs_time, plot_coverage_vs_speed, plot_l1_rsrp_cdf,
    plot_blockage_demo, plot_accuracy_vs_speed, plot_topk_bars
)


def ensure_data(speeds, n_ues, duration, seed):
    for speed in speeds:
        try:
            load_channels(speed, seed)
            print(f"  {speed} km/h: cached")
        except FileNotFoundError:
            print(f"  {speed} km/h: generating...")
            channels, distances = generate_sionna_channels(speed, n_ues, duration, seed)
            save_channels(channels, distances, speed, seed)


def evaluate_temporal(model, loader, device="cpu"):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = all_logits.argmax(1)
    top1 = (preds == all_labels).float().mean().item()

    _, top3_idx = all_logits.topk(3, dim=1)
    top3 = (top3_idx == all_labels.unsqueeze(1)).any(1).float().mean().item()

    _, top5_idx = all_logits.topk(5, dim=1)
    top5 = (top5_idx == all_labels.unsqueeze(1)).any(1).float().mean().item()

    return {
        "top1": top1, "top3": top3, "top5": top5,
        "predicted": preds.numpy(),
        "labels": all_labels.numpy(),
    }


def evaluate_last_beam(loader):
    """Last-beam baseline: predict same beam as last timestep in window."""
    all_preds = []
    all_labels = []

    for features, labels in loader:
        last_powers = features[:, -1, :]
        preds = last_powers.argmax(dim=1)
        wide_to_narrow = (preds.float() * (N_NARROW_BEAMS / N_WIDE_BEAMS)).long()
        wide_to_narrow = wide_to_narrow.clamp(0, N_NARROW_BEAMS - 1)
        all_preds.append(wide_to_narrow)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    top1 = (all_preds == all_labels).float().mean().item()
    return {"top1": top1, "predicted": all_preds.numpy(), "labels": all_labels.numpy()}


def run_single_speed(speed_kmh, device, seed, cross_speed_threshold=None):
    print(f"\n{'='*50}")
    print(f"  Speed: {speed_kmh} km/h")
    print(f"{'='*50}")

    t0 = time.time()
    print("\n[1] Loading temporal data...")
    loaders = get_temporal_dataloaders(speed_kmh, seed=seed)
    print(f"  Train: {len(loaders['train'].dataset)}, Cal: {len(loaders['cal'].dataset)}, "
          f"Val: {len(loaders['val'].dataset)}, Test: {len(loaders['test'].dataset)}")

    print("\n[2] Last-beam baseline...")
    lastbeam_results = evaluate_last_beam(loaders["test"])
    print(f"  Last-beam top-1: {lastbeam_results['top1']:.4f}")

    print("\n[3] Training BeamMLP (spatial-only)...")
    mlp_model = BeamMLP()
    mlp_model = train_temporal_model(loaders["train"], loaders["val"], device=device, model=mlp_model)
    mlp_results = evaluate_temporal(mlp_model, loaders["test"], device)
    print(f"  MLP top-1: {mlp_results['top1']:.4f} | top-3: {mlp_results['top3']:.4f} | top-5: {mlp_results['top5']:.4f}")

    print("\n[4] Training BeamLSTM...")
    lstm_model = BeamLSTM()
    lstm_model = train_temporal_model(loaders["train"], loaders["val"], device=device, model=lstm_model)
    lstm_results = evaluate_temporal(lstm_model, loaders["test"], device)
    print(f"  LSTM top-1: {lstm_results['top1']:.4f} | top-3: {lstm_results['top3']:.4f} | top-5: {lstm_results['top5']:.4f}")

    print("\n[5] Training BeamTemporalTransformer...")
    tf_model = BeamTemporalTransformer()
    tf_model = train_temporal_model(loaders["train"], loaders["val"], device=device, model=tf_model)
    tf_results = evaluate_temporal(tf_model, loaders["test"], device)
    print(f"  TF top-1: {tf_results['top1']:.4f} | top-3: {tf_results['top3']:.4f} | top-5: {tf_results['top5']:.4f}")

    models = {"MLP": (mlp_model, mlp_results), "LSTM": (lstm_model, lstm_results), "TF": (tf_model, tf_results)}
    best_name = max(models, key=lambda k: models[k][1]["top1"])
    best_model, best_results = models[best_name]
    print(f"\n  Best model: {best_name} (top-1: {best_results['top1']:.4f})")

    print("\n[6] Conformal prediction (per-speed calibration)...")
    threshold, cal_scores = calibrate(best_model, loaders["cal"],
                                      alpha=CONFORMAL_ALPHA, device=device)
    print(f"  Per-speed threshold: {threshold:.4f}")

    prediction_sets_list = predict_sets(best_model, loaders["test"], threshold, device=device)
    sizes = set_sizes(prediction_sets_list)
    cov = coverage(prediction_sets_list, loaders["test_labels"])
    print(f"  Per-speed coverage: {cov:.4f} (target: {1 - CONFORMAL_ALPHA:.2f})")
    print(f"  Mean set size: {sizes.mean():.2f}, Median: {np.median(sizes):.0f}")

    cross_cov = None
    cross_sizes = None
    if cross_speed_threshold is not None:
        cross_sets = predict_sets(best_model, loaders["test"], cross_speed_threshold, device=device)
        cross_sizes_arr = set_sizes(cross_sets)
        cross_cov = coverage(cross_sets, loaders["test_labels"])
        cross_sizes = cross_sizes_arr
        print(f"  Cross-speed coverage: {cross_cov:.4f}, Mean set size: {cross_sizes_arr.mean():.2f}")

    print(f"\n  Done in {time.time() - t0:.1f}s")

    return {
        "speed_kmh": speed_kmh,
        "lastbeam_results": lastbeam_results,
        "mlp_results": mlp_results,
        "lstm_results": lstm_results,
        "tf_results": tf_results,
        "best_name": best_name,
        "best_model": best_model,
        "best_results": best_results,
        "threshold": threshold,
        "prediction_sets": prediction_sets_list,
        "sizes": sizes,
        "coverage": cov,
        "cross_coverage": cross_cov,
        "cross_sizes": cross_sizes,
        "loaders": loaders,
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal beam prediction pipeline")
    parser.add_argument("--speed", type=int, default=None, help="Single speed (km/h)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ues", type=int, default=500)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-blockage", action="store_true")
    args = parser.parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    speeds = [args.speed] if args.speed else SPEEDS_KMH

    print("\n[0] Ensuring channel data exists...")
    ensure_data(speeds, args.n_ues, args.duration, args.seed)

    t_total = time.time()
    all_outputs = {}
    cross_threshold = None

    for speed in speeds:
        out = run_single_speed(speed, device, args.seed,
                               cross_speed_threshold=cross_threshold)
        all_outputs[speed] = out

        if speed == 30 or (args.speed and speed == args.speed):
            cross_threshold = out["threshold"]
            print(f"\n  Saved 30 km/h threshold ({cross_threshold:.4f}) for cross-speed experiment")

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")

    print(f"\n  {'Speed':>6} | {'Last-beam':>9} | {'MLP':>6} | {'LSTM':>6} | {'TF':>6} | {'CP cov':>6} | {'Set sz':>6}")
    print(f"  {'-'*6} | {'-'*9} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    for speed, out in all_outputs.items():
        print(f"  {speed:>4d}  | {out['lastbeam_results']['top1']:>9.4f} | "
              f"{out['mlp_results']['top1']:>6.4f} | {out['lstm_results']['top1']:>6.4f} | "
              f"{out['tf_results']['top1']:>6.4f} | {out['coverage']:>6.4f} | "
              f"{out['sizes'].mean():>6.2f}")

    if len(all_outputs) > 1:
        print("\n  Top-k accuracy (best model per speed):")
        for speed, out in all_outputs.items():
            r = out["best_results"]
            print(f"  {speed:>4d} km/h | top-1: {r['top1']:.4f} | top-3: {r['top3']:.4f} | top-5: {r['top5']:.4f}")

    oh_results = {}
    for speed, out in all_outputs.items():
        oh = overhead_slots(out["sizes"])
        oh_results[speed] = oh
        print(f"  {speed:>4d} km/h | Overhead reduction: {oh['mean_reduction']*100:.1f}%")

    if not args.no_blockage and 60 in all_outputs:
        print("\n[Blockage Demo]")
        out_60 = all_outputs[60]
        demo_result = run_blockage_demo(
            speed_kmh=60, model=out_60["best_model"],
            threshold=out_60["threshold"], seed=args.seed, device=device,
        )
        print(f"  Early warning lead time: {demo_result['lead_time_s']:.3f}s")
        plot_blockage_demo(demo_result)

    print("\n[Generating figures]")
    if len(all_outputs) > 1:
        plot_coverage_vs_speed(all_outputs)
        plot_accuracy_vs_speed(all_outputs)
        plot_topk_bars(all_outputs)

    print(f"\nTotal time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
