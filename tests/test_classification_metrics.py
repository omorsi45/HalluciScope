"""Classification metrics and the cluster bootstrap: expectations are derived by hand.

The shared literal is gold [S, C, I, S] against pred [S, S, I, C]. Pairing the units:
(S, S) is a true negative, (C, S) a false negative, (I, I) a true positive, (S, C) a false
positive. Positive means "not supported", so tp = fp = fn = tn = 1.
"""

import numpy as np
import pytest

from evaluation.metrics.bootstrap import cluster_bootstrap
from evaluation.metrics.classification import (
    accepted_risk,
    confusion,
    detection_prf,
    false_supported_rate,
    macro_f1,
    response_level,
    supported_coverage,
)

S, C, I = "supported", "contradicted", "insufficient"  # noqa: E741 (I is the class initial)
GOLD = [S, C, I, S]
PRED = [S, S, I, C]
TOL = 1e-12


def test_detection_literal():
    result = detection_prf(GOLD, PRED)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5, abs=TOL)
    assert result["recall"] == pytest.approx(0.5, abs=TOL)
    assert result["f1"] == pytest.approx(0.5, abs=TOL)


def test_detection_uses_not_supported_as_the_positive_class():
    # Hand count: pairs (C,I) tp, (I,I) tp, (S,C) fp, (C,S) fn, (I,S) fn, (S,S) tn.
    # precision = 2/3, recall = 2/4, F1 = 2*2 / (2*2 + 1 + 2) = 4/7. The wrong unsupported class
    # in (C,I) still counts as a detection.
    result = detection_prf([C, I, S, C, I, S], [I, I, C, S, S, S])
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 2, 1)
    assert result["precision"] == pytest.approx(2 / 3, abs=TOL)
    assert result["recall"] == pytest.approx(0.5, abs=TOL)
    assert result["f1"] == pytest.approx(4 / 7, abs=TOL)


def test_detection_zero_denominators_give_zero():
    # Nothing is gold-positive or predicted-positive, so precision, recall and F1 are all 0.0.
    result = detection_prf([S, S], [S, S])
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (0, 0, 0, 2)
    assert (result["precision"], result["recall"], result["f1"]) == (0.0, 0.0, 0.0)


def test_false_supported_and_risk_literal():
    # Two gold-not-supported units (C, I); one of them (the C) is predicted supported.
    assert false_supported_rate(GOLD, PRED) == pytest.approx(0.5, abs=TOL)
    # Two predicted-supported units (index 0 and 1); one of them (gold C) is gold-not-supported.
    assert accepted_risk(GOLD, PRED) == (0.5, 1, 2)
    # Three gold-not-supported units (C, I, I); two of them (index 0 and 1) are predicted supported.
    assert false_supported_rate([C, I, I, S], [S, S, I, I]) == pytest.approx(2 / 3, abs=TOL)
    # Three predicted-supported units (indices 0, 1, 3); two of them are gold not supported.
    assert accepted_risk([C, I, S, S], [S, S, I, S]) == (pytest.approx(2 / 3, abs=TOL), 2, 3)


def test_false_supported_and_risk_undefined_cases():
    # No gold-not-supported unit: the rate is undefined. No predicted-supported unit: risk is.
    assert false_supported_rate([S, S], [S, C]) is None
    assert accepted_risk([S, C], [C, I]) == (None, 0, 0)


def test_macro_f1_literal():
    # F1_S = 0.5 (tp 1, fp 1, fn 1), F1_C = 0 (gold C predicted S), F1_I = 1 (tp 1).
    assert macro_f1(GOLD, PRED) == pytest.approx(0.5, abs=TOL)


def test_absent_class_excluded():
    # No contradicted unit anywhere. F1_S: tp 1, fp 0, fn 1 -> 2/3. F1_I: tp 2, fp 1, fn 0 -> 4/5.
    # The mean is over those two classes only: (2/3 + 4/5) / 2 = 11/15 (not 22/45).
    gold = [S, I, S, I]
    pred = [S, I, I, I]
    assert macro_f1(gold, pred) == pytest.approx(11 / 15, abs=TOL)


def test_supported_coverage():
    assert supported_coverage(PRED) == pytest.approx(0.5, abs=TOL)
    assert supported_coverage([S, S, S, I]) == pytest.approx(0.75, abs=TOL)


def test_confusion_is_gold_by_predicted_with_every_class_present():
    assert confusion(GOLD, PRED) == {
        S: {S: 1, C: 1, I: 0},
        C: {S: 1, C: 0, I: 0},
        I: {S: 0, C: 0, I: 1},
    }
    # Not symmetric: two gold C predicted S, one gold S predicted I.
    assert confusion([C, C, S], [S, S, I]) == {
        S: {S: 0, C: 0, I: 1},
        C: {S: 2, C: 0, I: 0},
        I: {S: 0, C: 0, I: 0},
    }


def test_response_level():
    # A response is positive when any of its units is. Hand count over six responses:
    #   r1 gold [S, C] pred [S, S]  -> gold +, pred -   fn
    #   r2 gold [S, S] pred [S, C]  -> gold -, pred +   fp
    #   r3 gold [I]    pred [I]     -> both +           tp
    #   r4 gold [C, S] pred [S, C]  -> both +           tp
    #   r5 gold [S]    pred [S]     -> both -           tn
    #   r6 gold [C]    pred [S]     -> gold +, pred -   fn
    # tp 2, fp 1, fn 2, tn 1: precision 2/3, recall 1/2, F1 = 4 / (4 + 1 + 2) = 4/7.
    gold = [S, C, S, S, I, C, S, S, C]
    pred = [S, S, S, C, I, S, C, S, S]
    responses = ["r1", "r1", "r2", "r2", "r3", "r4", "r4", "r5", "r6"]
    result = response_level(gold, pred, responses)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 2, 1)
    assert result["precision"] == pytest.approx(2 / 3, abs=TOL)
    assert result["recall"] == pytest.approx(0.5, abs=TOL)
    assert result["f1"] == pytest.approx(4 / 7, abs=TOL)


def test_response_level_does_not_need_adjacent_units():
    # r1's units are split apart; its gold C still makes the whole response positive.
    result = response_level([C, S, S], [C, S, S], ["r1", "r2", "r1"])
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 0, 0, 1)


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        detection_prf([S, C], [S])
    with pytest.raises(ValueError):
        detection_prf([S, "maybe"], [S, S])
    with pytest.raises(ValueError):
        supported_coverage([S, "maybe"])
    # Length mismatch is rejected even when the metric would otherwise be undefined.
    with pytest.raises(ValueError):
        false_supported_rate([S, S], [S])
    with pytest.raises(ValueError):
        accepted_risk([S, S], [C])
    with pytest.raises(ValueError):
        response_level([S, C], [S, C], ["r1"])


def test_empty_inputs_give_zero_or_none():
    assert macro_f1([], []) == 0.0
    assert supported_coverage([]) == 0.0
    assert false_supported_rate([], []) is None
    assert accepted_risk([], []) == (None, 0, 0)


VALUES = np.array([0.0, 1.0, 1.0, 4.0, 5.0, 9.0, 2.0, 2.0])
GROUPS = ["a", "a", "b", "b", "b", "c", "d", "d"]


def _mean(indices: np.ndarray) -> float:
    return float(VALUES[indices].mean())


def test_bootstrap_deterministic_and_brackets():
    first = cluster_bootstrap(GROUPS, _mean, n_resamples=200)
    second = cluster_bootstrap(GROUPS, _mean, n_resamples=200)
    point, lo, hi = first
    assert first == second
    assert point == pytest.approx(VALUES.mean(), abs=TOL)
    assert lo is not None
    assert hi is not None
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_seed_is_used():
    default = cluster_bootstrap(GROUPS, _mean, n_resamples=200)
    other = cluster_bootstrap(GROUPS, _mean, n_resamples=200, seed=7)
    assert other[0] == default[0]
    assert other[1:] != default[1:]


def test_bootstrap_constant_statistic():
    constant = np.full(6, 0.5)  # 0.5 is exact in binary, so every resample mean is exactly 0.5
    point, lo, hi = cluster_bootstrap(
        ["a", "a", "b", "c", "c", "c"], lambda idx: float(constant[idx].mean())
    )
    assert lo == hi == point == 0.5


def test_bootstrap_resamples_whole_groups_with_multiplicity():
    groups = ["a", "a", "b", "c", "c", "c"]
    seen: list[np.ndarray] = []

    def record(indices: np.ndarray) -> float:
        seen.append(np.array(indices))
        return 0.0

    cluster_bootstrap(groups, record, n_resamples=50)
    # First call is the point estimate over every row, once.
    assert seen[0].tolist() == [0, 1, 2, 3, 4, 5]
    sizes = {"a": 2, "b": 1, "c": 3}
    members = {"a": [0, 1], "b": [2], "c": [3, 4, 5]}
    draw_counts = []
    for indices in seen[1:]:
        counts = {name: 0 for name in sizes}
        position = 0
        # Walk the resample group by group: each drawn group contributes all of its rows, in order.
        while position < len(indices):
            name = groups[int(indices[position])]
            block = indices[position : position + sizes[name]].tolist()
            assert block == members[name]
            counts[name] += 1
            position += sizes[name]
        # Three unique groups means three draws with replacement.
        assert sum(counts.values()) == 3
        draw_counts.append(counts)
    # With replacement: some resample misses a group and some repeats one.
    assert any(0 in counts.values() for counts in draw_counts)
    assert any(max(counts.values()) >= 2 for counts in draw_counts)


@pytest.mark.parametrize(
    ("ci", "lo_expected", "hi_expected"),
    [
        # Resample values are 1..100. numpy's linear percentile at q sits at 1 + q/100 * 99.
        (0.95, 1 + 0.025 * 99, 1 + 0.975 * 99),
        (0.5, 1 + 0.25 * 99, 1 + 0.75 * 99),
    ],
)
def test_bootstrap_percentiles_follow_ci(ci, lo_expected, hi_expected):
    values = iter([-1.0] + [float(v) for v in range(1, 101)])
    point, lo, hi = cluster_bootstrap(
        ["a", "b", "c"], lambda idx: next(values), n_resamples=100, ci=ci
    )
    assert point == -1.0
    assert lo == pytest.approx(lo_expected, abs=TOL)
    assert hi == pytest.approx(hi_expected, abs=TOL)


def test_bootstrap_skips_none_results():
    calls = {"n": 0}

    def flaky(indices: np.ndarray) -> float | None:
        calls["n"] += 1
        if calls["n"] == 1:
            return 2.0  # the point estimate
        return None if calls["n"] % 2 == 0 else 4.0

    assert cluster_bootstrap(["a", "b", "c"], flaky, n_resamples=10) == (2.0, 4.0, 4.0)


def test_bootstrap_all_none_returns_nones():
    assert cluster_bootstrap(["a", "b"], lambda idx: None, n_resamples=10) == (None, None, None)
    # A defined point estimate survives when every resample is undefined.
    calls = {"n": 0}

    def point_only(indices: np.ndarray) -> float | None:
        calls["n"] += 1
        return 3.0 if calls["n"] == 1 else None

    assert cluster_bootstrap(["a", "b"], point_only, n_resamples=10) == (3.0, None, None)
