import numpy as np
from beampred.codebook import get_narrow_codebook
from beampred.config import N_NARROW_BEAMS, N_WIDE_BEAMS, SNR_VALUES_DB
from beampred.utils import from_db


def adaptive_beam_management(prediction_sets, true_labels, test_channels,
                             confidence_threshold=3):
    narrow_cb = get_narrow_codebook()
    all_gains = np.abs(test_channels @ narrow_cb.conj().T) ** 2

    n = len(true_labels)
    selected_beams = np.zeros(n, dtype=int)
    overheads = np.zeros(n)
    used_ml = np.zeros(n, dtype=bool)

    for i in range(n):
        pset = prediction_sets[i]
        if len(pset) <= confidence_threshold:
            candidate_gains = all_gains[i, pset]
            selected_beams[i] = pset[np.argmax(candidate_gains)]
            overheads[i] = N_WIDE_BEAMS + len(pset)
            used_ml[i] = True
        else:
            selected_beams[i] = np.argmax(all_gains[i])
            overheads[i] = N_NARROW_BEAMS

    accuracy = np.mean(selected_beams == true_labels)
    ml_fraction = np.mean(used_ml)
    avg_overhead = np.mean(overheads)

    return {
        "selected_beams": selected_beams,
        "overheads": overheads,
        "accuracy": accuracy,
        "ml_fraction": ml_fraction,
        "avg_overhead": avg_overhead,
        "used_ml": used_ml,
    }


def sweep_thresholds(prediction_sets, true_labels, test_channels,
                     thresholds=None):
    if thresholds is None:
        thresholds = list(range(1, 11))

    narrow_cb = get_narrow_codebook()
    all_gains = np.abs(test_channels @ narrow_cb.conj().T) ** 2

    results = []
    for t in thresholds:
        r = adaptive_beam_management(prediction_sets, true_labels, test_channels, t)

        tp_results = {}
        for snr_db in SNR_VALUES_DB:
            snr_lin = from_db(snr_db)
            se_vals = np.log2(1 + snr_lin * all_gains[np.arange(len(true_labels)), r["selected_beams"]])
            frame_slots = 100
            data_fraction = (frame_slots - r["overheads"]) / frame_slots
            tp_vals = se_vals * data_fraction
            tp_results[snr_db] = np.mean(tp_vals)

        results.append({
            "threshold": t,
            "accuracy": r["accuracy"],
            "ml_fraction": r["ml_fraction"],
            "avg_overhead": r["avg_overhead"],
            "throughput": tp_results,
        })

    return results
