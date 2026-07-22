import numpy as np
import pytest

from cohortweave_eval.metrics import (
    c_index,
    concordance_correlation_coefficient,
    constraint_violation_rate,
    dice_binary,
    fidelity_preservation_rate,
    imbalance_ratio,
    imbalance_reduction_ratio,
    ks_distance,
    mean_prevalence_difference,
    relative_mae_improvement,
    wasserstein_reduction,
)


def test_ks_identity():
    assert ks_distance([1, 2, 3], [1, 2, 3]) == 0.0


def test_mean_prevalence_difference():
    # Reference A/B = 75/25%; candidate = 50/50%; mean absolute drift = 25%.
    assert mean_prevalence_difference(["A", "A", "A", "B"], ["A", "A", "B", "B"]) == pytest.approx(0.25)


def test_wasserstein_reduction():
    ref = np.array([0, 1, 2])
    shifted = ref + 10
    harmonised = ref + 1
    assert wasserstein_reduction(ref, shifted, harmonised) > 80


def test_ccc_identity():
    assert concordance_correlation_coefficient([1, 2, 3], [1, 2, 3]) == 1.0


def test_ccc_penalises_location_shift():
    assert concordance_correlation_coefficient([1, 2, 3], [2, 3, 4]) < 1.0


def test_dice():
    assert dice_binary([1, 1, 0], [1, 0, 0]) == 2 / 3


def test_imbalance_metrics():
    before = imbalance_ratio(["M"] * 2135 + ["F"] * 1000)
    after = imbalance_ratio(["M"] * 1000 + ["F"] * 1000)
    assert before == pytest.approx(2.135)
    assert after == pytest.approx(1.0)
    assert imbalance_reduction_ratio(before, after) == pytest.approx(53.1616, rel=1e-4)


def test_constraint_violation_rate():
    assert constraint_violation_rate([True, False, True, False]) == 50.0


def test_fidelity_preservation_rate():
    assert fidelity_preservation_rate([0.01, 0.04, 0.06, 0.0], threshold=0.05) == 75.0


def test_relative_mae_improvement():
    assert relative_mae_improvement(18.205, 12.344) == pytest.approx(32.1945, rel=1e-4)


def test_c_index_perfect():
    risk = np.array([3, 2, 1])
    time = np.array([1, 2, 3])
    event = np.array([1, 1, 1])
    assert c_index(risk, time, event) == 1.0
