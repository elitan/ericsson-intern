"""3GPP-compliant KPIs: L1-RSRP gap, overhead slots, effective throughput.

All metrics follow 3GPP TR 38.901 / TS 38.214 conventions.
- L1-RSRP: measured over wide+narrow beams, 20ms period
- Overhead: number of beam measurement slots consumed
- Throughput: accounts for overhead reduction from ML prediction
"""
import numpy as np

from beampred.config import (
    N_NARROW_BEAMS, N_WIDE_BEAMS, SAMPLE_INTERVAL_S
)


def l1_rsrp_gap(predicted_beams, true_beams, channel_gains_db):
    """L1-RSRP gap between predicted and oracle beams (dB).

    channel_gains_db: (n_samples, n_narrow_beams) — per-beam RSRP in dB.
    Returns per-sample gap and summary stats.
    """
    n = len(predicted_beams)
    pred_rsrp = channel_gains_db[np.arange(n), predicted_beams]
    oracle_rsrp = channel_gains_db[np.arange(n), true_beams]
    gap = oracle_rsrp - pred_rsrp
    gap = np.maximum(gap, 0.0)

    return {
        "gap_db": gap,
        "mean": np.mean(gap),
        "median": np.median(gap),
        "p90": np.percentile(gap, 90),
        "p95": np.percentile(gap, 95),
    }


def overhead_slots(prediction_set_sizes, n_wide=N_WIDE_BEAMS, n_narrow=N_NARROW_BEAMS):
    """Beam management overhead in slots.

    ML approach: n_wide (initial sweep) + set_size (refinement).
    Exhaustive: n_narrow slots.
    """
    ml_overhead = n_wide + prediction_set_sizes
    exhaustive_overhead = np.full_like(prediction_set_sizes, n_narrow)
    reduction = 1.0 - ml_overhead / exhaustive_overhead

    return {
        "ml_slots": ml_overhead,
        "exhaustive_slots": exhaustive_overhead,
        "reduction_frac": reduction,
        "mean_reduction": np.mean(reduction),
        "mean_ml_slots": np.mean(ml_overhead),
    }


def effective_throughput(se_per_sample, overhead_slots_per_sample, total_slots=100,
                         slot_duration_s=SAMPLE_INTERVAL_S):
    """Effective throughput accounting for overhead.

    se_per_sample: spectral efficiency in bits/s/Hz per sample.
    overhead_slots_per_sample: slots used for beam management.
    total_slots: total slots in measurement period.
    """
    data_fraction = np.maximum(1.0 - overhead_slots_per_sample / total_slots, 0.0)
    effective_se = se_per_sample * data_fraction

    return {
        "effective_se": effective_se,
        "mean_se": np.mean(effective_se),
        "data_fraction": data_fraction,
        "mean_data_fraction": np.mean(data_fraction),
    }


def compute_beam_rsrp(channels, codebook):
    """Compute per-beam RSRP in dB from channels and codebook.

    channels: (n_samples, n_tx) complex
    codebook: (n_beams, n_tx) complex
    Returns: (n_samples, n_beams) RSRP in dB
    """
    gains = np.abs(channels @ codebook.conj().T) ** 2
    return 10 * np.log10(np.maximum(gains, 1e-30))


def compute_all_kpis(predicted_beams, true_beams, channels, codebook,
                     prediction_set_sizes=None):
    """Compute all 3GPP KPIs in one call."""
    rsrp_db = compute_beam_rsrp(channels, codebook)

    rsrp_result = l1_rsrp_gap(predicted_beams, true_beams, rsrp_db)

    if prediction_set_sizes is not None:
        overhead_result = overhead_slots(prediction_set_sizes)
    else:
        overhead_result = overhead_slots(np.ones(len(predicted_beams), dtype=int))

    pred_rsrp_linear = 10 ** (rsrp_db[np.arange(len(predicted_beams)), predicted_beams] / 10)
    se = np.log2(1 + pred_rsrp_linear)

    tp_result = effective_throughput(se, overhead_result["ml_slots"])

    return {
        "l1_rsrp": rsrp_result,
        "overhead": overhead_result,
        "throughput": tp_result,
    }


def format_kpi_table(kpi_results, speed_kmh=None):
    """Format KPI results as a printable string."""
    lines = []
    prefix = f"[{speed_kmh} km/h] " if speed_kmh else ""

    rsrp = kpi_results["l1_rsrp"]
    oh = kpi_results["overhead"]
    tp = kpi_results["throughput"]

    lines.append(f"{prefix}L1-RSRP gap: mean={rsrp['mean']:.2f} dB, "
                 f"median={rsrp['median']:.2f} dB, p90={rsrp['p90']:.2f} dB")
    lines.append(f"{prefix}Overhead: mean={oh['mean_ml_slots']:.1f} slots, "
                 f"reduction={oh['mean_reduction']*100:.1f}%")
    lines.append(f"{prefix}Throughput: mean SE={tp['mean_se']:.3f} bits/s/Hz, "
                 f"data fraction={tp['mean_data_fraction']*100:.1f}%")

    return "\n".join(lines)
