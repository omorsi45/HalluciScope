"""Calibration metrics: every expected value below is derived by hand, not by the code under test.

Classes are always (supported, contradicted, insufficient); probability columns follow that
order and `y` holds class indices.
"""

import numpy as np
import pytest

from backend.calibration.metrics import (
    aurc,
    brier_multiclass,
    classwise_ece,
    ece_top_label,
    nll,
    reliability_bins,
    risk_coverage,
    softmax,
)

TOL = 1e-12


def _rows_with_top_confidence(confidences):
    """3-class rows whose largest probability sits in column 0 and equals the given value."""
    rest = [(1.0 - c) / 2 for c in confidences]
    return np.array([[c, r, r] for c, r in zip(confidences, rest, strict=True)])


def test_ece_literal():
    # Bin 13 holds the two 0.9 rows: accuracy 1/2, gap 0.4, weight 2/4 -> 0.2.
    # Bin 9 holds the two 0.6 rows: accuracy 2/2, gap 0.4, weight 2/4 -> 0.2. Total 0.4.
    probs = _rows_with_top_confidence([0.9, 0.9, 0.6, 0.6])
    y = np.array([0, 1, 0, 0])  # argmax is column 0 everywhere: correct = [1, 0, 1, 1]
    assert ece_top_label(probs, y) == pytest.approx(0.4, abs=TOL)


def test_perfect_calibration_zero():
    # Confidence 1.0 maps to bin index 15 before the clamp; it must land in the last bin.
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 1, 2])
    assert ece_top_label(probs, y) == 0.0


def test_top_label_ece_counts_wrong_argmax_as_incorrect():
    # One confident row, wrong: confidence 0.8, accuracy 0, so ECE = 0.8.
    probs = np.array([[0.1, 0.8, 0.1]])
    assert ece_top_label(probs, np.array([0])) == pytest.approx(0.8, abs=TOL)


def test_classwise_ece_literal():
    # n_bins=2 puts p < 0.5 in bin 0 and p >= 0.5 in bin 1.
    # Class 0: p = [.75, .75, .25, 0], hit = [1, 0, 0, 0]
    #   bin 1: conf .75, acc 1/2, gap .25, weight 2/4 -> .125
    #   bin 0: conf .125, acc 0, gap .125, weight 2/4 -> .0625             total .1875
    # Class 1: p = [.25, .25, .75, 0], hit = [0, 1, 1, 0]
    #   bin 1: conf .75, acc 1, gap .25, weight 1/4 -> .0625
    #   bin 0: conf 1/6, acc 1/3, gap 1/6, weight 3/4 -> .125              total .1875
    # Class 2: p = [0, 0, 0, 1], hit = [0, 0, 0, 1]: no gap in either bin   total 0
    # Mean over the 3 classes: (.1875 + .1875 + 0) / 3 = .125.
    probs = np.array(
        [
            [0.75, 0.25, 0.0],
            [0.75, 0.25, 0.0],
            [0.25, 0.75, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 2])
    assert classwise_ece(probs, y, n_bins=2) == pytest.approx(0.125, abs=TOL)


def test_brier_and_nll_literals():
    probs = np.array([[0.7, 0.2, 0.1]])
    y = np.array([0])
    # Brier: 0.3^2 + 0.2^2 + 0.1^2 = 0.14. NLL: -ln(0.7).
    assert brier_multiclass(probs, y) == pytest.approx(0.14, abs=TOL)
    assert nll(probs, y) == pytest.approx(0.35667494393873245, abs=TOL)


def test_brier_and_nll_average_over_rows():
    # Row 1 Brier 0.14; row 2 (true class 1): 0.1^2 + 0.9^2 + 0.8^2 = 1.46; mean 1.6 / 2 = 0.8.
    # NLL: (-ln 0.7 - ln 0.1) / 2 = (0.35667494393873245 + 2.302585092994046) / 2.
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    y = np.array([0, 1])
    assert brier_multiclass(probs, y) == pytest.approx(0.8, abs=TOL)
    assert nll(probs, y) == pytest.approx(1.329630018466389, abs=1e-9)


def test_nll_floors_a_zero_probability_at_eps():
    # The true class has probability 0: the loss is -ln(1e-12) = 12 ln 10, not infinity.
    probs = np.array([[0.0, 1.0, 0.0]])
    assert nll(probs, np.array([0])) == pytest.approx(27.631021115928547, abs=1e-9)
    assert nll(probs, np.array([0]), eps=1e-6) == pytest.approx(13.815510557964274, abs=1e-9)


def test_risk_coverage_literal():
    score = np.array([0.9, 0.8, 0.7, 0.6])
    gold_ok = np.array([1, 0, 1, 1])
    coverage, risks = risk_coverage(score, gold_ok)
    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0], atol=TOL)
    np.testing.assert_allclose(risks, [0.0, 0.5, 1 / 3, 0.25], atol=TOL)
    assert aurc(risks) == pytest.approx(0.2708333333333333, abs=TOL)


def test_risk_coverage_orders_by_descending_score():
    # Accepted in order 0.9 (ok), 0.8 (ok), 0.7 (bad), 0.6 (ok): risks 0, 0, 1/3, 1/4.
    score = np.array([0.6, 0.9, 0.7, 0.8])
    gold_ok = np.array([True, True, False, True])
    coverage, risks = risk_coverage(score, gold_ok)
    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0], atol=TOL)
    np.testing.assert_allclose(risks, [0.0, 0.0, 1 / 3, 0.25], atol=TOL)


def test_risk_coverage_breaks_ties_by_input_order():
    # Equal scores keep input order (stable sort): ok, ok, bad -> risks 0, 0, 1/3.
    # A reversed tie order would give bad, ok, ok -> risks 1, 1/2, 1/3.
    coverage, risks = risk_coverage(np.array([0.5, 0.5, 0.5]), np.array([1, 1, 0]))
    np.testing.assert_allclose(coverage, [1 / 3, 2 / 3, 1.0], atol=TOL)
    np.testing.assert_allclose(risks, [0.0, 0.0, 1 / 3], atol=TOL)


def test_reliability_bins_counts():
    # Bin edges at 15 bins are k/15: 0.05 -> 0, 0.10 and 0.12 -> 1, 0.50 and 0.52 -> 7,
    # 0.99 -> 14, and 1.0 is clamped into the last bin (14).
    conf = np.array([0.05, 0.10, 0.12, 0.50, 0.52, 0.99, 1.0])
    correct = np.array([1, 1, 0, 1, 1, 0, 1])
    bins = reliability_bins(conf, correct)
    assert len(bins) == 15
    assert [b["count"] for b in bins] == [1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2]
    assert sum(b["count"] for b in bins) == len(conf)


def test_reliability_bins_fields_and_empty_bins():
    # n_bins=4: [0, .25), [.25, .5), [.5, .75), [.75, 1]. Bin 2 is empty.
    conf = np.array([0.0, 0.3, 0.3, 1.0])
    correct = np.array([1, 0, 1, 1])
    bins = reliability_bins(conf, correct, n_bins=4)
    assert [b["lo"] for b in bins] == [0.0, 0.25, 0.5, 0.75]
    assert [b["hi"] for b in bins] == [0.25, 0.5, 0.75, 1.0]
    assert [b["count"] for b in bins] == [1, 2, 0, 1]
    assert bins[0]["mean_conf"] == pytest.approx(0.0, abs=TOL)
    assert bins[0]["accuracy"] == pytest.approx(1.0, abs=TOL)
    assert bins[1]["mean_conf"] == pytest.approx(0.3, abs=TOL)
    assert bins[1]["accuracy"] == pytest.approx(0.5, abs=TOL)
    assert bins[2]["mean_conf"] is None
    assert bins[2]["accuracy"] is None
    assert bins[3]["mean_conf"] == pytest.approx(1.0, abs=TOL)
    assert bins[3]["accuracy"] == pytest.approx(1.0, abs=TOL)


def test_softmax_stable():
    probs = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0, abs=TOL)
    np.testing.assert_allclose(probs, [[1.0, 0.0, 0.0]], atol=TOL)


def test_softmax_is_row_wise():
    # Row 0 is uniform; row 1 is [2, 1, 1] / 4 because exp(ln 2) = 2.
    probs = softmax(np.array([[0.0, 0.0, 0.0], [np.log(2.0), 0.0, 0.0]]))
    np.testing.assert_allclose(probs[0], [1 / 3, 1 / 3, 1 / 3], atol=TOL)
    np.testing.assert_allclose(probs[1], [0.5, 0.25, 0.25], atol=TOL)
