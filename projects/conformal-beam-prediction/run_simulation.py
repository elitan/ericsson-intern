import argparse
import time
import numpy as np
import torch
import copy

from beampred import config
from beampred.config import N_WIDE_BEAMS, N_NARROW_BEAMS, CONFORMAL_ALPHA, CONFIDENCE_THRESHOLD
from beampred.dataset import get_dataloaders
from beampred.train import train_model
from beampred.baselines import hierarchical_search, train_logistic_regression, predict_logistic, BeamCNN
from beampred.beam_predictor import BeamPredictor, ResNetMLP, BeamTransformer
from beampred.conformal import (calibrate, predict_sets, set_sizes, coverage,
                                 calibrate_beam_aware, predict_sets_beam_aware,
                                 calibrate_group, predict_sets_group)
from beampred.complexity import analyze_complexity, print_complexity_table
from beampred.visualize import generate_all_figures
from beampred.evaluate import compute_metrics, compute_metrics_from_predictions, print_summary
from beampred.error_analysis import run_error_analysis
from beampred.adaptive_fallback import sweep_thresholds
from beampred.export_results import export_latex_tables, format_results_summary


METHOD_ORDER = ["MLP", "ResNet-MLP", "CNN", "Transformer", "LogReg", "Hierarchical"]


def run_single_seed(seed, source, device, use_cache, t_total, scenario="O1_28"):
    config.SEED = seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    def step_time(label, t0):
        print(f"  [{label} done in {time.time() - t0:.1f}s, total {time.time() - t_total:.1f}s]")

    t0 = time.time()
    print(f"\n[1/13] Loading dataset (seed={seed}, scenario={scenario})...")
    src = source
    try:
        loaders = get_dataloaders(seed=seed, use_cache=use_cache, source=src, scenario=scenario)
    except Exception as e:
        print(f"  {src} failed ({e}), falling back to synthetic")
        src = "synthetic"
        loaders = get_dataloaders(seed=seed, use_cache=use_cache, source=src, scenario=scenario)

    train_loader, cal_loader, val_loader, test_loader, test_channels, test_dist, cal_dist, test_labels, mean, std = loaders
    print(f"  Source: {src}")
    print(f"  Train: {len(train_loader.dataset)}, Cal: {len(cal_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    step_time("1/13", t0)

    t0 = time.time()
    print("\n[2/13] Training MLP...")
    mlp_model = train_model(train_loader, val_loader, device=device)
    step_time("2/13", t0)

    t0 = time.time()
    print("\n[3/13] Training ResNet-MLP...")
    resnet_model = ResNetMLP()
    resnet_model = train_model(train_loader, val_loader, device=device, model=resnet_model)
    step_time("3/13", t0)

    t0 = time.time()
    print("\n[4/13] Training CNN...")
    cnn_model = BeamCNN()
    cnn_model = train_model(train_loader, val_loader, device=device, model=cnn_model)
    step_time("4/13", t0)

    t0 = time.time()
    print("\n[5/13] Training Transformer...")
    transformer_model = BeamTransformer()
    transformer_model = train_model(train_loader, val_loader, device=device, model=transformer_model)
    step_time("5/13", t0)

    t0 = time.time()
    print("\n[6/13] Training logistic regression...")
    train_feat_np = np.vstack([b[0].numpy() for b in train_loader])
    train_labels_np = np.concatenate([b[1].numpy() for b in train_loader])
    test_feat_np = np.vstack([b[0].numpy() for b in test_loader])
    test_labels_np_loader = np.concatenate([b[1].numpy() for b in test_loader])

    lr_model = train_logistic_regression(train_feat_np, train_labels_np)
    lr_predicted = predict_logistic(lr_model, test_feat_np)
    print(f"  LogReg top-1: {np.mean(lr_predicted == test_labels_np_loader):.4f}")
    step_time("6/13", t0)

    t0 = time.time()
    print("\n[7/13] Running hierarchical beam search...")
    hier_predicted, hier_overhead = hierarchical_search(test_channels)
    print(f"  Hierarchical top-1: {np.mean(hier_predicted == np.array(test_labels)):.4f}, overhead: {hier_overhead}")
    step_time("7/13", t0)

    t0 = time.time()
    print("\n[8/13] Evaluating all methods...")
    all_results = []

    mlp_results = compute_metrics(mlp_model, test_loader, test_channels, test_dist,
                                  test_labels, device=device, overhead=N_WIDE_BEAMS, method_name="MLP")
    print_summary(mlp_results)
    all_results.append(mlp_results)

    resnet_results = compute_metrics(resnet_model, test_loader, test_channels, test_dist,
                                     test_labels, device=device, overhead=N_WIDE_BEAMS, method_name="ResNet-MLP")
    print_summary(resnet_results)
    all_results.append(resnet_results)

    cnn_results = compute_metrics(cnn_model, test_loader, test_channels, test_dist,
                                  test_labels, device=device, overhead=N_WIDE_BEAMS, method_name="CNN")
    print_summary(cnn_results)
    all_results.append(cnn_results)

    transformer_results = compute_metrics(transformer_model, test_loader, test_channels, test_dist,
                                          test_labels, device=device, overhead=N_WIDE_BEAMS, method_name="Transformer")
    print_summary(transformer_results)
    all_results.append(transformer_results)

    lr_results = compute_metrics_from_predictions(lr_predicted, test_labels, test_channels,
                                                   test_dist, N_WIDE_BEAMS, method_name="LogReg")
    print_summary(lr_results)
    all_results.append(lr_results)

    hier_results = compute_metrics_from_predictions(hier_predicted, test_labels, test_channels,
                                                     test_dist, hier_overhead, method_name="Hierarchical")
    print_summary(hier_results)
    all_results.append(hier_results)
    step_time("8/13", t0)

    model_map = {"MLP": mlp_model, "ResNet-MLP": resnet_model,
                  "CNN": cnn_model, "Transformer": transformer_model}
    nn_results = [mlp_results, resnet_results, cnn_results, transformer_results]
    best_idx = max(range(len(nn_results)), key=lambda i: nn_results[i]["top1"])
    best_name = nn_results[best_idx]["method"]
    best_model = model_map[best_name]

    t0 = time.time()
    print(f"\n[9/16] Conformal prediction (standard) on {best_name}...")
    threshold, cal_scores = calibrate(best_model, cal_loader, alpha=CONFORMAL_ALPHA, device=device)
    print(f"  Conformal threshold: {threshold:.4f}")
    prediction_sets = predict_sets(best_model, test_loader, threshold, device=device)
    sizes = set_sizes(prediction_sets)
    cov = coverage(prediction_sets, test_labels)
    print(f"  Coverage: {cov:.4f} (target: {1 - CONFORMAL_ALPHA:.2f})")
    print(f"  Mean set size: {sizes.mean():.2f}, Median: {np.median(sizes):.0f}")

    print(f"\n  Conformal prediction (beam-aware) on {best_name}...")
    ba_threshold, ba_scores = calibrate_beam_aware(best_model, cal_loader, alpha=CONFORMAL_ALPHA, device=device)
    print(f"  Beam-aware threshold: {ba_threshold:.4f}")
    ba_prediction_sets = predict_sets_beam_aware(best_model, test_loader, ba_threshold, device=device)
    ba_sizes = set_sizes(ba_prediction_sets)
    ba_cov = coverage(ba_prediction_sets, test_labels)
    print(f"  Beam-aware coverage: {ba_cov:.4f}")
    print(f"  Beam-aware mean set size: {ba_sizes.mean():.2f}")
    step_time("9/16", t0)

    t0 = time.time()
    print(f"\n[10/16] Alpha sweep on {best_name}...")
    alpha_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    alpha_sweep_results = {"alphas": alpha_values, "std": [], "beam_aware": []}
    for alpha_val in alpha_values:
        th_std, _ = calibrate(best_model, cal_loader, alpha=alpha_val, device=device)
        ps_std = predict_sets(best_model, test_loader, th_std, device=device)
        cov_std = coverage(ps_std, test_labels)
        sz_std = set_sizes(ps_std).mean()

        th_ba, _ = calibrate_beam_aware(best_model, cal_loader, alpha=alpha_val, device=device)
        ps_ba = predict_sets_beam_aware(best_model, test_loader, th_ba, device=device)
        cov_ba = coverage(ps_ba, test_labels)
        sz_ba = set_sizes(ps_ba).mean()

        alpha_sweep_results["std"].append({"alpha": alpha_val, "coverage": cov_std, "mean_set_size": sz_std})
        alpha_sweep_results["beam_aware"].append({"alpha": alpha_val, "coverage": cov_ba, "mean_set_size": sz_ba})
        print(f"  alpha={alpha_val:.2f}  std: cov={cov_std:.4f} sz={sz_std:.2f}  ba: cov={cov_ba:.4f} sz={sz_ba:.2f}")
    step_time("10/16", t0)

    t0 = time.time()
    print(f"\n[10b/16] Top-K baseline comparison...")
    best_model.eval()
    with torch.no_grad():
        all_logits = []
        for feat_batch, _ in test_loader:
            logits = best_model(feat_batch.to(device))
            all_logits.append(logits.cpu())
        all_logits = torch.cat(all_logits)
    topk_baseline = []
    for k in [1, 2, 3, 5]:
        topk_preds = torch.topk(all_logits, k, dim=1).indices
        topk_correct = (topk_preds == torch.tensor(test_labels).unsqueeze(1)).any(dim=1).float()
        topk_cov = topk_correct.mean().item()
        topk_baseline.append({"k": k, "coverage": topk_cov, "set_size": k})
        print(f"  Top-{k}: coverage={topk_cov:.4f}, set_size={k}")
    print(f"  CP (α=0.1): coverage={cov:.4f}, avg_set_size={sizes.mean():.2f}")
    step_time("10b/16", t0)

    t0 = time.time()
    print(f"\n[11/16] Conditional coverage by distance quartile...")
    test_dist_arr = np.array(test_dist)
    quartiles = np.percentile(test_dist_arr, [25, 50, 75])
    bin_edges = [test_dist_arr.min(), quartiles[0], quartiles[1], quartiles[2], test_dist_arr.max()]
    bin_labels_cc = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}m" for i in range(4)]
    conditional_coverage = []
    for i in range(4):
        mask = (test_dist_arr >= bin_edges[i]) & (test_dist_arr < bin_edges[i+1] + (1 if i == 3 else 0))
        idx_bin = np.where(mask)[0]
        if len(idx_bin) == 0:
            continue
        ps_bin = [prediction_sets[j] for j in idx_bin]
        labels_bin = [test_labels[j] for j in idx_bin]
        cov_bin = coverage(ps_bin, labels_bin)
        sz_bin = set_sizes(ps_bin).mean()
        ba_ps_bin = [ba_prediction_sets[j] for j in idx_bin]
        ba_cov_bin = coverage(ba_ps_bin, labels_bin)
        ba_sz_bin = set_sizes(ba_ps_bin).mean()
        conditional_coverage.append({
            "bin_label": bin_labels_cc[i], "count": len(idx_bin),
            "std_coverage": cov_bin, "std_set_size": sz_bin,
            "ba_coverage": ba_cov_bin, "ba_set_size": ba_sz_bin,
        })
        print(f"  {bin_labels_cc[i]:>12s} (n={len(idx_bin):>4d}): std cov={cov_bin:.4f} sz={sz_bin:.2f}  ba cov={ba_cov_bin:.4f} sz={ba_sz_bin:.2f}")
    step_time("11/16", t0)

    t0 = time.time()
    print(f"\n[12/16] Group-conditional CP on {best_name}...")
    cal_dist_arr = np.array(cal_dist)
    group_thresholds, group_bin_edges = calibrate_group(cal_scores, cal_dist_arr, alpha=CONFORMAL_ALPHA, n_bins=4)
    best_model.eval()
    with torch.no_grad():
        all_probs = []
        for feat_batch, _ in test_loader:
            feat_batch = feat_batch.to(device)
            logits = best_model(feat_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        all_probs = np.vstack(all_probs)
    group_prediction_sets = predict_sets_group(all_probs, test_dist_arr, group_thresholds, group_bin_edges)
    group_sizes = set_sizes(group_prediction_sets)
    group_cov = coverage(group_prediction_sets, test_labels)
    print(f"  Group-CP coverage: {group_cov:.4f}, mean set size: {group_sizes.mean():.2f}")
    group_conditional_coverage = []
    for i in range(4):
        mask = (test_dist_arr >= bin_edges[i]) & (test_dist_arr < bin_edges[i+1] + (1 if i == 3 else 0))
        idx_bin = np.where(mask)[0]
        if len(idx_bin) == 0:
            continue
        ps_bin = [group_prediction_sets[j] for j in idx_bin]
        labels_bin = [test_labels[j] for j in idx_bin]
        cov_bin = coverage(ps_bin, labels_bin)
        sz_bin = set_sizes(ps_bin).mean()
        std_cov_bin = conditional_coverage[i]["std_coverage"] if i < len(conditional_coverage) else 0
        group_conditional_coverage.append({
            "bin_label": bin_labels_cc[i], "count": len(idx_bin),
            "std_coverage": std_cov_bin, "group_coverage": cov_bin, "group_set_size": sz_bin,
        })
        print(f"  {bin_labels_cc[i]:>12s}: std={std_cov_bin:.4f} -> group={cov_bin:.4f} sz={sz_bin:.2f}")
    step_time("12/16", t0)

    t0 = time.time()
    print("\n[13/16] Adaptive fallback...")
    sweep_results = sweep_thresholds(prediction_sets, np.array(test_labels), test_channels)
    for sr in sweep_results:
        print(f"  Threshold={sr['threshold']:>2d}: acc={sr['accuracy']:.4f}, "
              f"ML={sr['ml_fraction']:.3f}, overhead={sr['avg_overhead']:.1f}")
    step_time("13/16", t0)

    t0 = time.time()
    print("\n[14/16] Error analysis...")
    best_results = nn_results[best_idx]
    error_results = run_error_analysis(best_results["predicted"], best_results["labels"], test_channels)
    print(f"  Cost-weighted score: {error_results['cost_weighted_score']:.4f}")
    loss = error_results["gain_loss_per_distance"]
    counts = error_results["gain_loss_counts"]
    for k in range(min(5, len(loss))):
        if counts[k] > 0:
            print(f"  Off-by-{k+1}: {loss[k]:.2f} dB loss (n={counts[k]})")
    step_time("14/16", t0)

    t0 = time.time()
    print("\n[15/16] Complexity analysis...")
    models_dict = {
        "MLP": mlp_model,
        "ResNet-MLP": resnet_model,
        "CNN": cnn_model,
        "Transformer": transformer_model,
    }
    complexity_results = analyze_complexity(models_dict)
    print_complexity_table(complexity_results)
    step_time("15/16", t0)

    return {
        "all_results": all_results,
        "best_results": best_results,
        "error_results": error_results,
        "sweep_results": sweep_results,
        "sizes": sizes,
        "ba_sizes": ba_sizes,
        "group_sizes": group_sizes,
        "coverage": cov,
        "ba_coverage": ba_cov,
        "group_coverage": group_cov,
        "complexity": complexity_results,
        "alpha_sweep": alpha_sweep_results,
        "conditional_coverage": conditional_coverage,
        "group_conditional_coverage": group_conditional_coverage,
        "topk_baseline": topk_baseline,
    }


def aggregate_seeds(seed_outputs):
    n_methods = len(seed_outputs[0]["all_results"])
    mean_results = []
    std_results = []

    for m in range(n_methods):
        method_name = seed_outputs[0]["all_results"][m]["method"]
        keys = ["top1", "top3", "top5"]
        mean_r = {"method": method_name, "overhead": seed_outputs[0]["all_results"][m]["overhead"]}
        std_r = {}

        for k in keys:
            vals = [so["all_results"][m][k] for so in seed_outputs]
            mean_r[k] = np.mean(vals)
            std_r[k] = np.std(vals)

        snr_mean = {}
        snr_std = {}
        for snr in config.SNR_VALUES_DB:
            for sk in ["se_exhaustive", "se_method", "se_ratio", "tp_exhaustive", "tp_method"]:
                vals = [so["all_results"][m]["snr_results"][snr][sk] for so in seed_outputs]
                if snr not in snr_mean:
                    snr_mean[snr] = {}
                    snr_std[snr] = {}
                snr_mean[snr][sk] = np.mean(vals)
                snr_std[snr][sk] = np.std(vals)
        mean_r["snr_results"] = snr_mean
        std_r["snr_results"] = snr_std

        dist_mean = {}
        dist_std = {}
        ref_dist = seed_outputs[0]["all_results"][m]["distance_results"]
        for b in ref_dist:
            vals = [so["all_results"][m]["distance_results"].get(b, {}).get("top1", 0) for so in seed_outputs]
            dist_mean[b] = {"top1": np.mean(vals), "range": ref_dist[b]["range"],
                            "count": ref_dist[b]["count"]}
            dist_std[b] = {"top1": np.std(vals)}
        mean_r["distance_results"] = dist_mean
        std_r["distance_results"] = dist_std

        mean_r["confusion"] = seed_outputs[0]["all_results"][m]["confusion"]
        mean_r["predicted"] = seed_outputs[0]["all_results"][m]["predicted"]
        mean_r["labels"] = seed_outputs[0]["all_results"][m]["labels"]

        mean_results.append(mean_r)
        std_results.append(std_r)

    return mean_results, std_results


def main():
    parser = argparse.ArgumentParser(description="mmWave beam prediction simulator")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--source", type=str, default="deepmimo", choices=["deepmimo", "synthetic"])
    parser.add_argument("--scenario", type=str, default="O1_28", help="DeepMIMO scenario (e.g. O1_28, I3_28)")
    parser.add_argument("--temporal", action="store_true", help="Run temporal pipeline instead")
    args = parser.parse_args()

    if args.temporal:
        from run_temporal import main as temporal_main
        import sys
        sys.argv = [sys.argv[0]]
        if args.epochs:
            sys.argv.extend(["--epochs", str(args.epochs)])
        temporal_main()
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.epochs is not None:
        config.EPOCHS = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Scenario: {args.scenario}")

    t_total = time.time()
    seed_outputs = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        out = run_single_seed(seed, args.source, device, not args.no_cache, t_total, scenario=args.scenario)
        seed_outputs.append(out)

    mean_results, std_results = aggregate_seeds(seed_outputs)

    last = seed_outputs[-1]
    complexity_results = last["complexity"]

    print(f"\n{'='*60}")
    print("  AGGREGATED RESULTS ({} seeds)".format(len(seeds)))
    print(f"{'='*60}")

    aggregated_for_export = []
    for mr, sr in zip(mean_results, std_results):
        aggregated_for_export.append((mr["method"], mr, sr))

    print(format_results_summary(aggregated_for_export))

    cov_vals = [so["coverage"] for so in seed_outputs]
    ba_cov_vals = [so["ba_coverage"] for so in seed_outputs]
    group_cov_vals = [so["group_coverage"] for so in seed_outputs]
    size_vals = [so["sizes"].mean() for so in seed_outputs]
    ba_size_vals = [so["ba_sizes"].mean() for so in seed_outputs]
    group_size_vals = [so["group_sizes"].mean() for so in seed_outputs]
    print(f"\nConformal coverage (standard): {np.mean(cov_vals):.4f} ± {np.std(cov_vals):.4f}")
    print(f"Conformal coverage (beam-aware): {np.mean(ba_cov_vals):.4f} ± {np.std(ba_cov_vals):.4f}")
    print(f"Conformal coverage (group-CP): {np.mean(group_cov_vals):.4f} ± {np.std(group_cov_vals):.4f}")
    print(f"Mean set size (standard): {np.mean(size_vals):.2f} ± {np.std(size_vals):.2f}")
    print(f"Mean set size (beam-aware): {np.mean(ba_size_vals):.2f} ± {np.std(ba_size_vals):.2f}")
    print(f"Mean set size (group-CP): {np.mean(group_size_vals):.2f} ± {np.std(group_size_vals):.2f}")

    print("\n[16/16] Generating figures...")
    t0 = time.time()

    generate_all_figures(
        all_results=mean_results,
        mlp_results=mean_results[0],
        error_results=last["error_results"],
        sweep_results=last["sweep_results"],
        set_sizes_arr=last["sizes"],
        std_results=std_results if len(seeds) > 1 else None,
        set_sizes_beam_aware=last["ba_sizes"],
        alpha_sweep=last["alpha_sweep"],
        conditional_coverage=last["conditional_coverage"],
        group_conditional_coverage=last["group_conditional_coverage"],
    )
    print(f"  [16/16 done in {time.time() - t0:.1f}s]")

    export_latex_tables(aggregated_for_export, complexity_results)

    print(f"\nDone. Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
