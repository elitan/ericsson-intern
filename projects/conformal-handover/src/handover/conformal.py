"""
Conformal prediction for handover.

Implements:
- Split conformal prediction (standard)
- Adaptive Conformal Inference (ACI) for temporal data
"""

import numpy as np


def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Calibrate conformal threshold using APS score.

    Args:
        probs: (n_cal, n_classes) softmax probabilities
        labels: (n_cal,) true labels
        alpha: target miscoverage rate (1-alpha = coverage)

    Returns:
        threshold q_hat
    """
    n = len(labels)
    scores = 1 - probs[np.arange(n), labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, q_level, method="higher")
    return q_hat


def predict_sets(
    probs: np.ndarray,
    threshold: float,
) -> list:
    """
    Generate prediction sets using calibrated threshold.

    Args:
        probs: (n_test, n_classes) softmax probabilities
        threshold: calibrated threshold

    Returns:
        list of prediction sets (arrays of class indices)
    """
    sets = []
    for p in probs:
        included = np.where(p >= 1 - threshold)[0]
        if len(included) == 0:
            included = np.array([p.argmax()])
        sets.append(included)
    return sets


def evaluate_cp(
    prediction_sets: list,
    labels: np.ndarray,
) -> dict:
    """Evaluate conformal prediction performance."""
    coverages = [labels[i] in prediction_sets[i] for i in range(len(labels))]
    sizes = [len(s) for s in prediction_sets]

    return {
        "coverage": np.mean(coverages),
        "avg_set_size": np.mean(sizes),
        "set_size_std": np.std(sizes),
        "size_1_frac": np.mean([s == 1 for s in sizes]),
    }


class GroupConditionalCP:
    """
    Group-conditional conformal prediction.

    Calibrates separate thresholds per group to ensure coverage
    guarantees hold within each subpopulation.
    """

    def __init__(self, n_groups: int, alpha: float = 0.1):
        self.n_groups = n_groups
        self.alpha = alpha
        self.thresholds = {}

    def calibrate(self, probs: np.ndarray, labels: np.ndarray, groups: np.ndarray):
        """Calibrate threshold per group."""
        for g in range(self.n_groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            g_probs = probs[mask]
            g_labels = labels[mask]
            n = len(g_labels)
            scores = 1 - g_probs[np.arange(n), g_labels]
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.thresholds[g] = np.quantile(scores, min(q_level, 1.0), method="higher")

    def predict(self, probs: np.ndarray, groups: np.ndarray) -> list:
        """Generate prediction sets using group-specific thresholds."""
        sets = []
        for i, (p, g) in enumerate(zip(probs, groups)):
            threshold = self.thresholds.get(g, 0.5)
            included = np.where(p >= 1 - threshold)[0]
            if len(included) == 0:
                included = np.array([p.argmax()])
            sets.append(included)
        return sets


def assign_speed_groups(speeds: np.ndarray, n_groups: int = 4) -> np.ndarray:
    """Assign samples to speed-based groups using quantiles."""
    quantiles = np.percentile(speeds, np.linspace(0, 100, n_groups + 1)[1:-1])
    groups = np.zeros(len(speeds), dtype=int)
    for i, q in enumerate(quantiles):
        groups[speeds >= q] = i + 1
    return groups


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) for temporal data.

    Updates threshold online to maintain coverage despite distribution shift.
    Based on Gibbs & Candes (2021).
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """
        Args:
            alpha: target miscoverage rate
            gamma: learning rate for threshold adaptation
        """
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha

    def reset(self):
        self.alpha_t = self.alpha

    def update(self, covered: bool):
        """Update effective alpha based on whether last prediction covered."""
        err_t = 1 - int(covered)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.999)

    def get_threshold(self, cal_scores: np.ndarray) -> float:
        """Get current threshold based on adaptive alpha."""
        n = len(cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha_t)) / n
        q_level = np.clip(q_level, 0, 1)
        return np.quantile(cal_scores, q_level, method="higher")


def calibrate_raps(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    k_reg: int = 1,
    lam: float = 0.01,
    rand: bool = True,
) -> float:
    """
    Calibrate RAPS (Regularized Adaptive Prediction Sets).

    RAPS penalizes large prediction sets, encouraging smaller sets.
    Based on Angelopoulos et al. (2020).

    Args:
        probs: (n_cal, n_classes) softmax probabilities
        labels: (n_cal,) true labels
        alpha: target miscoverage rate
        k_reg: classes before penalty kicks in (usually 1)
        lam: regularization strength (higher = smaller sets)
        rand: whether to use randomization

    Returns:
        calibrated threshold
    """
    n = len(labels)
    n_classes = probs.shape[1]

    scores = np.zeros(n)
    for i in range(n):
        sorted_idx = np.argsort(probs[i])[::-1]
        sorted_probs = probs[i][sorted_idx]

        true_rank = np.where(sorted_idx == labels[i])[0][0]

        cumsum = np.cumsum(sorted_probs)
        score = cumsum[true_rank]

        reg_penalty = lam * max(0, true_rank + 1 - k_reg)
        score += reg_penalty

        if rand:
            score += np.random.uniform(0, 1) * sorted_probs[true_rank]

        scores[i] = score

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, q_level, method="higher")
    return q_hat


def predict_sets_raps(
    probs: np.ndarray,
    threshold: float,
    k_reg: int = 1,
    lam: float = 0.01,
    rand: bool = True,
) -> list:
    """
    Generate prediction sets using RAPS.

    Args:
        probs: (n_test, n_classes) softmax probabilities
        threshold: calibrated RAPS threshold
        k_reg: regularization parameter
        lam: regularization strength
        rand: whether to use randomization

    Returns:
        list of prediction sets
    """
    sets = []
    for p in probs:
        sorted_idx = np.argsort(p)[::-1]
        sorted_probs = p[sorted_idx]

        cumsum = np.cumsum(sorted_probs)

        included = []
        for j, (idx, prob, cs) in enumerate(zip(sorted_idx, sorted_probs, cumsum)):
            score = cs + lam * max(0, j + 1 - k_reg)
            if rand:
                score += np.random.uniform(0, 1) * prob

            if score <= threshold:
                included.append(idx)
            else:
                included.append(idx)
                break

        if len(included) == 0:
            included = [p.argmax()]

        sets.append(np.array(included))
    return sets


def run_aci_online(
    probs: np.ndarray,
    labels: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.01,
) -> dict:
    """
    Run ACI on sequential data.

    Args:
        probs: (n_test, n_classes) softmax probabilities (in temporal order)
        labels: (n_test,) true labels
        cal_scores: (n_cal,) nonconformity scores from calibration set
        alpha: target miscoverage
        gamma: ACI learning rate

    Returns:
        dict with coverage, set sizes, per-step results
    """
    aci = AdaptiveConformalInference(alpha=alpha, gamma=gamma)

    coverages = []
    sizes = []
    thresholds = []

    for t in range(len(probs)):
        threshold = aci.get_threshold(cal_scores)
        thresholds.append(threshold)

        pred_set = np.where(probs[t] >= 1 - threshold)[0]
        if len(pred_set) == 0:
            pred_set = np.array([probs[t].argmax()])

        covered = labels[t] in pred_set
        coverages.append(covered)
        sizes.append(len(pred_set))

        aci.update(covered)

    return {
        "coverage": np.mean(coverages),
        "avg_set_size": np.mean(sizes),
        "coverages": np.array(coverages),
        "sizes": np.array(sizes),
        "thresholds": np.array(thresholds),
    }
