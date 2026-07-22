from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def ks_distance(reference, candidate) -> float:
    """Two-sample Kolmogorov–Smirnov statistic."""
    return float(ks_2samp(np.asarray(reference), np.asarray(candidate)).statistic)


def mean_prevalence_difference(reference, candidate) -> float:
    """Mean absolute category-prevalence difference as a unit fraction.

    Categories observed in either sample are included. Multiply by 100 for
    percentage points, matching the frozen Spark reports.
    """
    ref = list(reference)
    cand = list(candidate)
    if not ref or not cand:
        raise ValueError("reference and candidate must be non-empty")
    ref_counts = Counter(ref)
    cand_counts = Counter(cand)
    levels = set(ref_counts) | set(cand_counts)
    return float(
        np.mean(
            [
                abs(ref_counts[level] / len(ref) - cand_counts[level] / len(cand))
                for level in levels
            ]
        )
    )


def wasserstein_reduction(reference, shifted, harmonised) -> float:
    """Percentage reduction in 1-D Wasserstein distance after harmonisation."""
    before = float(wasserstein_distance(reference, shifted))
    after = float(wasserstein_distance(reference, harmonised))
    if before <= 0:
        return 0.0
    return 100.0 * (1.0 - after / before)


def concordance_correlation_coefficient(x, y) -> float:
    """Lin's concordance correlation coefficient (CCC)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be non-empty arrays with identical shape")
    vx = float(np.var(x))
    vy = float(np.var(y))
    mx = float(np.mean(x))
    my = float(np.mean(y))
    cov = float(np.mean((x - mx) * (y - my)))
    den = vx + vy + (mx - my) ** 2
    return 1.0 if den == 0 and np.allclose(x, y) else (0.0 if den == 0 else 2.0 * cov / den)


def dice_binary(mask_a, mask_b) -> float:
    """Sørensen–Dice overlap for two binary masks."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks must have identical shape")
    den = int(a.sum() + b.sum())
    return 1.0 if den == 0 else float(2 * np.logical_and(a, b).sum() / den)


def imbalance_ratio(values: Iterable[object]) -> float:
    """Largest observed subgroup count divided by the smallest."""
    counts = Counter(values)
    if len(counts) < 2:
        raise ValueError("at least two observed groups are required")
    observed = np.asarray(list(counts.values()), dtype=float)
    return float(observed.max() / observed.min())


def imbalance_reduction_ratio(imbalance_before: float, imbalance_after: float) -> float:
    """Percentage reduction in a ratio-valued imbalance measure."""
    if imbalance_before <= 0 or imbalance_after <= 0:
        raise ValueError("imbalance ratios must be positive")
    return float(100.0 * (1.0 - imbalance_after / imbalance_before))


def constraint_violation_rate(record_has_violation: Iterable[bool]) -> float:
    """Percentage of records with at least one rule violation."""
    values = np.asarray(list(record_has_violation), dtype=bool)
    if values.size == 0:
        raise ValueError("at least one record is required")
    return float(100.0 * values.mean())


def fidelity_preservation_rate(drifts: Iterable[float], threshold: float = 0.05) -> float:
    """Percentage of variables whose absolute drift is within threshold.

    Drifts and threshold are unit fractions; use 0.05 for five percentage points.
    """
    values = np.abs(np.asarray(list(drifts), dtype=float))
    if values.size == 0:
        raise ValueError("at least one drift value is required")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return float(100.0 * np.mean(values <= threshold))


def relative_mae_improvement(mae_baseline: float, mae_model: float) -> float:
    """Percentage MAE improvement versus a positive baseline MAE."""
    if mae_baseline <= 0 or mae_model < 0:
        raise ValueError("MAE values must be non-negative and baseline must be positive")
    return float(100.0 * (mae_baseline - mae_model) / mae_baseline)


def c_index(risk, time, event) -> float:
    """Harrell's concordance index for higher-risk-is-earlier-event scores."""
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    if not (risk.shape == time.shape == event.shape):
        raise ValueError("risk, time and event must have identical shape")
    num = den = 0.0
    for i in range(len(time)):
        if event[i] == 1:
            comp = time > time[i]
            den += int(comp.sum())
            num += int((risk[i] > risk[comp]).sum()) + 0.5 * int((risk[i] == risk[comp]).sum())
    return float("nan") if den == 0 else float(num / den)
