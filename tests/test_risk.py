"""Learn-then-Test risk control with Hoeffding-Bentkus p-values (spec ruling R1).

Expected values are hand-derived literals (checked against exact rational arithmetic), never
computed by the code under test. Classes are always (supported, contradicted, insufficient).
"""

import itertools

import numpy as np
import pytest

from backend.calibration.risk import (
    GRID,
    binom_cdf,
    clopper_pearson_upper,
    hb_pvalue,
    ltt_fixed_sequence,
    per_source_risk,
    select_thresholds,
)

SEED = 20261026


def test_grid_runs_from_095_down_to_050_in_steps_of_001():
    assert len(GRID) == 46
    assert GRID[0] == 0.95
    assert GRID[-1] == 0.5
    assert GRID[:3] == (0.95, 0.94, 0.93)
    assert all(a > b for a, b in itertools.pairwise(GRID))
    assert all(t == round(t, 2) for t in GRID)


def test_binom_cdf_literals():
    # sum_{i<=2} C(10,i) 0.1^i 0.9^(10-i) = 0.3486784401 + 0.3874204890 + 0.1937102445
    assert binom_cdf(2, 10, 0.1) == pytest.approx(0.9298091736000003, abs=1e-12)
    # P(Bin(20, 0.05) = 0) = 0.95^20
    assert binom_cdf(0, 20, 0.05) == pytest.approx(0.3584859224085419, abs=1e-12)
    # 0.5 + 0.5 * C(1000, 500) / 2^1000, exact rational arithmetic; large n must not underflow
    assert binom_cdf(500, 1000, 0.5) == pytest.approx(0.5126125090891804, abs=1e-12)


def test_binom_cdf_edge_cases():
    assert binom_cdf(-1, 10, 0.3) == 0.0
    assert binom_cdf(10, 10, 0.3) == 1.0
    assert binom_cdf(11, 10, 0.3) == 1.0
    assert binom_cdf(0, 10, 0.0) == 1.0
    assert binom_cdf(9, 10, 1.0) == 0.0


def test_clopper_pearson_literal():
    # k = 0: the largest p with (1 - p)^10 >= 0.05 is 1 - 0.05^(1/10)
    upper = clopper_pearson_upper(0, 10, 0.05)
    assert upper == pytest.approx(0.2588655508930523, abs=1e-9)
    # An upper bound errs high: the bisection must return the end of the bracket that is too large.
    assert upper >= 0.2588655508930523


def test_clopper_pearson_degenerate_inputs():
    assert clopper_pearson_upper(10, 10, 0.05) == 1.0
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0


def test_hb_pvalue_literals():
    # r_hat = 0: Hoeffding 0.9^100 beats Bentkus e * 0.9^100
    assert hb_pvalue(0.0, 100, 0.1) == pytest.approx(2.6561398887587334e-05, abs=1e-15)
    # r_hat = 0.05: Bentkus e * P(Bin(100, 0.1) <= 5) = 0.1565 beats Hoeffding 0.1881
    assert hb_pvalue(0.05, 100, 0.1) == pytest.approx(0.15651020427695356, abs=1e-12)
    # r_hat above alpha: no evidence at all
    assert hb_pvalue(0.12, 100, 0.1) == 1.0


def test_hb_pvalue_without_sources_is_one():
    assert hb_pvalue(0.0, 0, 0.1) == 1.0


def test_ceil_guard():
    # 0.07 * 100 = 7.000000000000001 in floats, so an unguarded ceil would use k = 8, giving
    # Bentkus 0.872 and a Hoeffding-only result of 0.5754. The guarded k = 7 gives
    # e * P(Bin(100, 0.1) <= 7) = 0.5601043133801665 (exact rational arithmetic).
    assert 0.07 * 100 > 7.0
    assert hb_pvalue(0.07, 100, 0.1) == pytest.approx(0.5601043133801665, abs=1e-12)


def test_per_source_risk_literal():
    # Rows interleaved on purpose. A: bad 0.9 and 0.2 (plus a high-scoring good row that must be
    # ignored), B: bad 0.95, C: only good rows (not a source in the risk).
    groups = np.array(["A", "B", "C", "A", "A", "C"])
    scores = np.array([0.9, 0.95, 0.99, 0.2, 0.97, 0.6])
    bad = np.array([True, True, False, True, False, False])
    # L_A = (1 + 0) / 2 = 0.5, L_B = 1, mean = 0.75 over 2 sources
    risk, n = per_source_risk(scores, bad, groups, 0.5)
    assert risk == pytest.approx(0.75)
    assert n == 2


def test_per_source_risk_accepts_score_equal_to_threshold():
    groups = np.array([1, 1])
    scores = np.array([0.5, 0.49])
    bad = np.array([True, True])
    risk, n = per_source_risk(scores, bad, groups, 0.5)
    assert risk == pytest.approx(0.5)
    assert n == 1


def test_per_source_risk_accept_mask_keeps_masked_rows_in_the_denominator():
    # A: bad 0.9 (mask on) and 0.8 (mask off); B: bad 0.95 (mask off).
    # Accepted needs score >= t AND mask, so L_A = 1/2 and L_B = 0, still two sources.
    groups = np.array(["A", "A", "B"])
    scores = np.array([0.9, 0.8, 0.95])
    bad = np.array([True, True, True])
    mask = np.array([True, False, False])
    risk, n = per_source_risk(scores, bad, groups, 0.5, accept_mask=mask)
    assert risk == pytest.approx(0.25)
    assert n == 2


def test_per_source_risk_without_bad_rows_has_no_sources():
    groups = np.array([1, 2])
    scores = np.array([0.9, 0.8])
    bad = np.array([False, False])
    assert per_source_risk(scores, bad, groups, 0.5)[1] == 0


_P_PASS = 2.6561398887587334e-05  # hb_pvalue(0.0, 100, 0.1), see test_hb_pvalue_literals


def test_fixed_sequence_stops_at_first_failure():
    table = {0.9: (0.0, 100), 0.8: (0.0, 100), 0.7: (0.2, 100), 0.6: (0.0, 100)}
    calls: list[float] = []

    def risk_at(t: float) -> tuple[float, int]:
        calls.append(t)
        return table[t]

    tau, trace = ltt_fixed_sequence((0.9, 0.8, 0.7, 0.6), risk_at, alpha=0.1, delta=0.05)

    assert tau == 0.8
    assert len(trace) == 3
    assert calls == [0.9, 0.8, 0.7]  # 0.6 would pass on its own but is never reached
    assert [row["t"] for row in trace] == [0.9, 0.8, 0.7]
    assert [row["r_hat"] for row in trace] == [0.0, 0.0, 0.2]
    assert [row["n"] for row in trace] == [100, 100, 100]
    assert trace[0]["p"] == pytest.approx(_P_PASS, abs=1e-15)
    assert trace[1]["p"] == pytest.approx(_P_PASS, abs=1e-15)
    assert trace[2]["p"] == 1.0


def test_fixed_sequence_returns_last_grid_point_when_nothing_fails():
    tau, trace = ltt_fixed_sequence((0.9, 0.8), lambda t: (0.0, 100), alpha=0.1, delta=0.05)
    assert tau == 0.8
    assert len(trace) == 2


def test_first_failure_returns_none():
    tau, trace = ltt_fixed_sequence((0.9, 0.8), lambda t: (0.5, 100), alpha=0.1, delta=0.05)
    assert tau is None
    assert len(trace) == 1
    assert trace[0]["t"] == 0.9


def _ordering_dataset(n_sources: int, *, leak_label: int | None = None):
    """Every source has a gold-supported, a gold-contradicted and a gold-insufficient row.

    The gold-supported row has p_supported exactly 0.40 and p_contradicted 0.60, so it is
    accepted as supported at tau_s = 0.40 and would be a false contradicted accept at t = 0.60 if
    the supported acceptance were not excluded. With `leak_label` (1 or 2), each source also gets
    a row of that gold class that scores 0.99 for supported, so SUPPORTED fails at the first grid
    point.
    """
    rows = [
        ([0.40, 0.60, 0.00], 0),
        ([0.02, 0.97, 0.01], 1),
        ([0.10, 0.05, 0.85], 2),
    ]
    if leak_label is not None:
        rows.append(([0.99, 0.005, 0.005], leak_label))
    probs = np.array([p for _ in range(n_sources) for p, _y in rows])
    y = np.array([label for _ in range(n_sources) for _p, label in rows])
    groups = np.repeat(np.arange(n_sources), len(rows))
    return probs, y, groups


def test_select_thresholds_ordering_excludes_rows_accepted_as_supported():
    probs, y, groups = _ordering_dataset(100)

    result = select_thresholds(probs, y, groups, grid=(0.9, 0.6, 0.4))

    assert result.tau_s == 0.4
    assert result.n_sources_s == 100
    assert len(result.trace_s) == 3
    # The gold-supported row is accepted as supported (0.40 >= tau_s), so it can never count as an
    # accepted contradiction, and CONTRADICTED certifies the whole grid. Without the exclusion it
    # would be accepted at t = 0.60 (L_i = 1/2) and the walk would stop at 0.9.
    assert result.tau_c == 0.4
    assert result.n_sources_c == 100
    assert len(result.trace_c) == 3
    assert [row["r_hat"] for row in result.trace_c] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("leak_label", [1, 2], ids=["gold-contradicted", "gold-insufficient"])
def test_select_thresholds_does_not_exclude_anything_without_a_supported_threshold(leak_label):
    probs, y, groups = _ordering_dataset(100, leak_label=leak_label)

    result = select_thresholds(probs, y, groups, grid=(0.9, 0.6, 0.4))

    # Any gold-not-supported row accepted as supported (0.99 >= 0.9) is a false-supported unit.
    assert result.tau_s is None
    assert len(result.trace_s) == 1
    # With no tau_s nothing is accepted as supported, so the gold-supported row (p_contradicted
    # 0.60) is a real false contradicted accept at t = 0.60 and the walk stops after 0.9.
    assert result.tau_c == 0.9
    assert len(result.trace_c) == 2


def test_select_thresholds_gold_insufficient_accepted_as_contradicted_blocks_tau_c():
    # Per source: a gold-insufficient row that scores 0.95 for contradicted (a false contradicted
    # accept at t = 0.9) and a gold-supported row that never scores for contradicted.
    rows = [([0.05, 0.95, 0.00], 2), ([0.90, 0.05, 0.05], 0)]
    probs = np.array([p for _ in range(100) for p, _y in rows])
    y = np.array([label for _ in range(100) for _p, label in rows])
    groups = np.repeat(np.arange(100), len(rows))

    result = select_thresholds(probs, y, groups, grid=(0.9, 0.6, 0.4))

    assert result.tau_s == 0.4  # the only gold-not-supported row scores 0.05 for supported
    # L_i = 1/2 at t = 0.9 (one of the two gold-not-contradicted rows is accepted)
    assert result.tau_c is None
    assert result.trace_c[0]["r_hat"] == pytest.approx(0.5)
    assert result.n_sources_c == 100


def test_select_thresholds_without_bad_rows_certifies_nothing():
    probs = np.array([[0.9, 0.05, 0.05], [0.95, 0.03, 0.02]])
    y = np.array([0, 0])
    groups = np.array([1, 2])

    result = select_thresholds(probs, y, groups, grid=(0.9, 0.6))

    assert result.tau_s is None
    assert result.n_sources_s == 0


ALPHA_S = 0.10
N_SOURCES = 300
N_REPETITIONS = 200
N_FRESH = 200_000


def _sample_calibration_set(rng: np.random.Generator):
    """300 sources with 1 to 3 bad rows (score ~ Beta(2, 5)) and 1 to 3 good rows (~ Beta(5, 2))."""
    n_bad = rng.integers(1, 4, size=N_SOURCES)
    n_good = rng.integers(1, 4, size=N_SOURCES)
    bad_scores = rng.beta(2, 5, size=int(n_bad.sum()))
    good_scores = rng.beta(5, 2, size=int(n_good.sum()))
    scores = np.concatenate([bad_scores, good_scores])
    # The supported score is the draw; the remaining mass is split evenly over the other classes.
    probs = np.column_stack([scores, (1 - scores) / 2, (1 - scores) / 2])
    # Bad rows are gold contradicted or insufficient (never supported), good rows are supported.
    y = np.concatenate([rng.integers(1, 3, size=bad_scores.size), np.zeros(good_scores.size, int)])
    groups = np.concatenate(
        [np.repeat(np.arange(N_SOURCES), n_bad), np.repeat(np.arange(N_SOURCES), n_good)]
    )
    return probs, y, groups


def test_ltt_guarantee_simulation():
    rng = np.random.default_rng(SEED)
    # True false-supported risk at each grid threshold: P(score >= t) for a fresh bad row. A
    # source's L_i is a mean of such indicators, so its expectation is the same value.
    fresh_bad = rng.beta(2, 5, size=N_FRESH)
    true_risk = {t: float(np.mean(fresh_bad >= t)) for t in GRID}
    # The scenario must be non-vacuous: the lowest threshold violates the target, so a procedure
    # that never stops would be caught.
    assert true_risk[GRID[-1]] > ALPHA_S

    violations = 0
    for _ in range(N_REPETITIONS):
        probs, y, groups = _sample_calibration_set(rng)
        tau_s = select_thresholds(probs, y, groups).tau_s
        if tau_s is not None and true_risk[tau_s] > ALPHA_S:
            violations += 1

    # Guarantee: P(risk(tau_s) > alpha_s) <= delta = 0.05; allow 0.03 for Monte Carlo noise.
    assert violations / N_REPETITIONS <= 0.05 + 0.03
