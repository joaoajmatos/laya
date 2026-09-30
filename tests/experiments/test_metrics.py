"""T004: metrics (experiments/metrics.py), against hand-computed values."""
import math

import numpy as np
import pytest

from experiments import metrics as M


def test_accuracy_is_exact_match_share():
    assert M.accuracy([1, 0, 1, 1]) == 0.75
    assert M.accuracy([]) is None


def test_level_error_is_mean_absolute_difference():
    assert M.level_error([0, 2, 3], [1, 2, 1]) == pytest.approx((1 + 0 + 2) / 3)


def test_ece_from_predicted_answer_probability():
    # bin 0.9-1.0: conf 0.95, 0.9 -> mean 0.925, accuracy 1/2 -> gap 0.425 (2 items)
    # bin 0.5-0.6: conf 0.55 -> accuracy 1 -> gap 0.45 (1 item)
    conf = [0.95, 0.9, 0.55]
    hit = [1, 0, 1]
    assert M.ece(conf, hit) == pytest.approx((2 * 0.425 + 1 * 0.45) / 3)


def test_ece_puts_probability_one_in_the_last_bin():
    assert M.ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    assert M.ece([1.0, 1.0], [0, 0]) == pytest.approx(1.0)


def test_ece_uses_answer_probability_not_entropy():
    # A uniform distribution has maximal entropy, but the predicted answer's probability is 1/k.
    # Confidence 1/3 with 1/3 accuracy is perfectly calibrated in its bin.
    assert M.ece([1 / 3] * 3, [1, 0, 0]) == pytest.approx(0.0, abs=1e-9)
    assert M.ece([], []) is None


def test_temperature_is_one_for_calibrated_data():
    # Probabilities match the empirical frequencies, so no rescaling helps.
    vecs, gold = [], []
    for i in range(100):
        vecs.append([0.8, 0.2])
        gold.append(0 if i % 5 else 1)   # 80% zeros
    assert M.fit_temperature(vecs, gold) == pytest.approx(1.0, abs=0.02)


def test_temperature_above_one_for_overconfident_data():
    vecs = [[0.99, 0.01]] * 100
    gold = [0 if i % 5 else 1 for i in range(100)]   # only 80% right
    t = M.fit_temperature(vecs, gold)
    assert t > 1.5
    assert M.fit_temperature([], []) == 1.0
    p = M.scale_probs([0.99, 0.01], t)
    assert p[0] == pytest.approx(0.8, abs=0.03)


def test_bootstrap_recovers_a_known_paired_difference():
    n = 200
    cases = [i // 5 for i in range(n)]
    a = np.ones(n)
    b = np.ones(n)
    b[np.random.default_rng(5).permutation(n)[: n // 4]] = 0   # b is right 75% of the time, irregularly
    out = M.paired_cluster_bootstrap(a, b, cases, n_boot=2000, seed=1)
    assert out["diff"] == pytest.approx(0.25)
    assert out["n_cases"] == 40 and out["n_items"] == 200
    assert out["lo"] <= out["diff"] <= out["hi"]
    again = M.paired_cluster_bootstrap(a, b, cases, n_boot=2000, seed=1)
    assert again == out                                   # reproducible for a fixed seed
    other = M.paired_cluster_bootstrap(a, b, cases, n_boot=2000, seed=2)
    assert other["lo"] <= other["diff"] <= other["hi"]   # another seed still brackets the difference


def test_bootstrap_interval_is_wider_when_a_cases_questions_are_perfectly_correlated():
    rng = np.random.default_rng(0)
    n_cases, per = 40, 5
    case_effect = rng.random(n_cases) < 0.5
    a = np.repeat(case_effect, per).astype(float)            # all five questions right or all wrong
    b = np.zeros(n_cases * per)
    cases = np.repeat(np.arange(n_cases), per)
    clustered = M.paired_cluster_bootstrap(a, b, cases, n_boot=3000, seed=3)
    naive = M.paired_cluster_bootstrap(a, b, np.arange(n_cases * per), n_boot=3000, seed=3)
    assert (clustered["hi"] - clustered["lo"]) > 1.5 * (naive["hi"] - naive["lo"])


def test_bootstrap_single_condition_and_empty():
    out = M.paired_cluster_bootstrap([1, 1, 0, 1], None, ["a", "a", "b", "b"], n_boot=200, seed=0)
    assert out["diff"] == 0.75
    assert M.paired_cluster_bootstrap([], None, [])["diff"] is None
    with pytest.raises(ValueError):
        M.paired_cluster_bootstrap([1, 0], [1], ["a", "b"])


@pytest.mark.parametrize("lo,hi,expected", [
    (0.01, 0.06, "better"),
    (-0.06, -0.01, "worse"),
    (-0.015, 0.015, "equal"),
    (-0.05, 0.03, "inconclusive"),
    (None, 0.1, "inconclusive"),
])
def test_verdict_rule(lo, hi, expected):
    assert M.verdict(lo, hi) == expected
