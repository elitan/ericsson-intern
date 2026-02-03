import numpy as np
from beampred.codebook import get_narrow_codebook
from beampred.config import N_NARROW_BEAMS


def beam_error_distances(predicted, true_labels):
    return np.abs(predicted.astype(int) - true_labels.astype(int))


def error_distance_histogram(predicted, true_labels):
    dists = beam_error_distances(predicted, true_labels)
    wrong = dists > 0
    if wrong.sum() == 0:
        return np.zeros(N_NARROW_BEAMS, dtype=int)
    hist, _ = np.histogram(dists[wrong], bins=np.arange(1, N_NARROW_BEAMS + 1))
    return hist


def gain_loss_per_distance(predicted, true_labels, test_channels, max_dist=20):
    narrow_cb = get_narrow_codebook()
    all_gains = np.abs(test_channels @ narrow_cb.conj().T) ** 2
    all_gains_db = 10 * np.log10(np.maximum(all_gains, 1e-30))

    dists = beam_error_distances(predicted, true_labels)
    mean_loss = np.zeros(max_dist)
    counts = np.zeros(max_dist, dtype=int)

    for k in range(1, max_dist + 1):
        mask = dists == k
        if mask.sum() == 0:
            continue
        optimal_gain = all_gains_db[mask, true_labels[mask]]
        pred_gain = all_gains_db[mask, predicted[mask]]
        mean_loss[k - 1] = np.mean(optimal_gain - pred_gain)
        counts[k - 1] = mask.sum()

    return mean_loss, counts


def cost_weighted_score(predicted, true_labels, test_channels):
    narrow_cb = get_narrow_codebook()
    all_gains = np.abs(test_channels @ narrow_cb.conj().T) ** 2

    optimal_gain = all_gains[np.arange(len(true_labels)), true_labels]
    pred_gain = all_gains[np.arange(len(predicted)), predicted]

    ratios = pred_gain / np.maximum(optimal_gain, 1e-30)
    return np.mean(ratios)


def run_error_analysis(predicted, true_labels, test_channels):
    hist = error_distance_histogram(predicted, true_labels)
    mean_loss, counts = gain_loss_per_distance(predicted, true_labels, test_channels)
    cws = cost_weighted_score(predicted, true_labels, test_channels)

    return {
        "error_histogram": hist,
        "gain_loss_per_distance": mean_loss,
        "gain_loss_counts": counts,
        "cost_weighted_score": cws,
    }
