import numpy as np
import torch
from beampred.config import N_NARROW_BEAMS, N_WIDE_BEAMS, SNR_VALUES_DB, DISTANCE_BINS, D_MIN, D_MAX
from beampred.codebook import get_narrow_codebook
from beampred.utils import from_db


def predict_beams(model, test_loader, device="cpu"):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def top_k_accuracy(logits, labels, k):
    topk = torch.topk(logits, k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def spectral_efficiency_batch(channels, beam_indices, codebook, snr_linear):
    norms = np.linalg.norm(channels, axis=1, keepdims=True)
    h_norm = channels / np.maximum(norms, 1e-30) * np.sqrt(channels.shape[1])
    gains = np.abs(np.sum(codebook[beam_indices].conj() * h_norm, axis=1)) ** 2
    return np.log2(1 + snr_linear * gains)


def spectral_efficiency(channel, beam_idx, codebook, snr_linear):
    h_norm = channel / (np.linalg.norm(channel) + 1e-30) * np.sqrt(len(channel))
    gain = np.abs(codebook[beam_idx].conj() @ h_norm) ** 2
    return np.log2(1 + snr_linear * gain)


def compute_metrics(model, test_loader, test_channels, test_distances,
                    test_labels, device="cpu", overhead=N_WIDE_BEAMS, method_name="AI"):
    logits, labels = predict_beams(model, test_loader, device)
    predicted = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()
    test_labels_np = np.array(test_labels)

    return _build_results(predicted, labels_np, test_labels_np, logits, labels,
                          test_channels, test_distances, overhead, method_name)


def compute_metrics_from_predictions(predicted, test_labels, test_channels,
                                     test_distances, overhead, method_name="method"):
    test_labels_np = np.array(test_labels)
    predicted = np.array(predicted)

    logits_dummy = torch.zeros(len(predicted), N_NARROW_BEAMS)
    logits_dummy[torch.arange(len(predicted)), torch.from_numpy(predicted)] = 10.0
    labels_t = torch.from_numpy(test_labels_np.copy())

    return _build_results(predicted, test_labels_np, test_labels_np, logits_dummy,
                          labels_t, test_channels, test_distances, overhead, method_name)


def _build_results(predicted, labels_np, test_labels_np, logits, labels,
                   test_channels, test_distances, overhead, method_name):
    narrow_cb = get_narrow_codebook()

    top1 = np.mean(predicted == labels_np)
    top3 = top_k_accuracy(logits, labels, 3) if logits is not None else top1
    top5 = top_k_accuracy(logits, labels, 5) if logits is not None else top1

    results = {
        "method": method_name,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "overhead": overhead,
        "snr_results": {},
        "distance_results": {},
        "predicted": predicted,
        "labels": labels_np,
        "confusion": np.zeros((N_NARROW_BEAMS, N_NARROW_BEAMS), dtype=int),
    }

    np.add.at(results["confusion"], (labels_np, predicted), 1)

    overhead_exhaustive = N_NARROW_BEAMS
    frame_slots = 100
    data_frac_exhaust = (frame_slots - overhead_exhaustive) / frame_slots
    data_frac_method = (frame_slots - overhead) / frame_slots

    for snr_db in SNR_VALUES_DB:
        snr_lin = from_db(snr_db)
        se_exhaustive = spectral_efficiency_batch(test_channels, test_labels_np, narrow_cb, snr_lin)
        se_method = spectral_efficiency_batch(test_channels, predicted, narrow_cb, snr_lin)

        mean_se_exh = np.mean(se_exhaustive)
        mean_se_mtd = np.mean(se_method)

        results["snr_results"][snr_db] = {
            "se_exhaustive": mean_se_exh,
            "se_method": mean_se_mtd,
            "se_ratio": mean_se_mtd / mean_se_exh if mean_se_exh > 0 else 0,
            "tp_exhaustive": mean_se_exh * data_frac_exhaust,
            "tp_method": mean_se_mtd * data_frac_method,
        }

    bin_edges = np.linspace(D_MIN, D_MAX, DISTANCE_BINS + 1)
    for b in range(DISTANCE_BINS):
        mask = (test_distances >= bin_edges[b]) & (test_distances < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        bin_pred = predicted[mask]
        bin_true = labels_np[mask]
        results["distance_results"][b] = {
            "range": (bin_edges[b], bin_edges[b + 1]),
            "top1": np.mean(bin_pred == bin_true),
            "count": int(mask.sum()),
        }

    return results


def print_summary(results):
    name = results.get("method", "Model")
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Top-1 accuracy: {results['top1']:.4f}")
    print(f"  Top-3 accuracy: {results['top3']:.4f}")
    print(f"  Top-5 accuracy: {results['top5']:.4f}")
    print(f"  Overhead: {results['overhead']} slots")
    print()

    print(f"  {'SNR':>6} {'SE_exh':>8} {'SE_mtd':>8} {'ratio':>8} {'TP_exh':>8} {'TP_mtd':>8}")
    print(f"  {'-'*50}")
    for snr_db in SNR_VALUES_DB:
        r = results["snr_results"][snr_db]
        print(
            f"  {snr_db:>6} {r['se_exhaustive']:>8.3f} {r['se_method']:>8.3f} "
            f"{r['se_ratio']:>8.3f} {r['tp_exhaustive']:>8.3f} {r['tp_method']:>8.3f}"
        )

    if results["distance_results"]:
        print()
        print("  Distance-binned top-1:")
        for b, dr in sorted(results["distance_results"].items()):
            print(f"    [{dr['range'][0]:.0f}-{dr['range'][1]:.0f}m] {dr['top1']:.4f} (n={dr['count']})")
