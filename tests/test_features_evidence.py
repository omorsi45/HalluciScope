import math

import numpy as np

from backend.verify.features import FEATURE_NAMES, compute_features, order_signals
from backend.verify.types import EvidenceSignal, NLIResult

CLAIM = "Refunds take 30 days"


def _signal(passage_id, text, nli, rerank, dense_cos):
    return EvidenceSignal(
        passage_id=passage_id, text=text, nli=nli, rerank=rerank, dense_cos=dense_cos
    )


def _s1():
    return _signal("p1", "Refunds take 30 days.", NLIResult(0.9, 0.05, 0.05), 2.0, 0.8)


def _s2():
    return _signal("p2", "Shipping is free.", NLIResult(0.1, 0.2, 0.7), -1.0, 0.3)


def _feature(name, vector):
    return vector[FEATURE_NAMES.index(name)]


def test_feature_names_length_12():
    assert len(FEATURE_NAMES) == 12
    assert FEATURE_NAMES == (
        "max_entail",
        "max_contra",
        "entail_minus_contra",
        "neutral_at_best",
        "n_entail_ge_half",
        "top_rerank",
        "max_dense_cos",
        "number_coverage",
        "has_numbers",
        "content_coverage",
        "claim_len",
        "contra_and_entail",
    )


def test_feature_vector_literal():
    vector = compute_features(CLAIM, [_s1(), _s2()], evidence_k=4)

    expected = [
        0.9,
        0.2,
        0.7,
        0.05,
        math.log1p(1),
        0.8807970779778823,
        0.8,
        1.0,
        1.0,
        1.0,
        1.6094379124341003,
        0.2,
    ]
    assert vector.dtype == np.float64
    assert vector.shape == (12,)
    np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-12)


def test_evidence_k_truncates_by_rerank():
    # s2 is listed first but ranks second by rerank, so evidence_k=1 keeps only s1.
    vector = compute_features(CLAIM, [_s2(), _s1()], evidence_k=1)

    assert _feature("max_contra", vector) == 0.05


def test_number_mismatch_coverage():
    vector = compute_features("Refunds take 14 days", [_s1()], evidence_k=4)

    assert _feature("number_coverage", vector) == 0.0
    assert _feature("has_numbers", vector) == 1.0


def test_money_and_percent_numbers():
    money = _signal("p1", "The fee is 1200 dollars.", NLIResult(0.8, 0.1, 0.1), 1.0, 0.5)
    percent = _signal("p2", "Growth was 5% last year.", NLIResult(0.8, 0.1, 0.1), 1.0, 0.5)
    bare = _signal("p3", "Growth was 5 last year.", NLIResult(0.8, 0.1, 0.1), 1.0, 0.5)

    fee = compute_features("The fee is $1,200", [money], evidence_k=4)
    growth = compute_features("Growth was 5%", [percent], evidence_k=4)
    no_percent = compute_features("Growth was 5%", [bare], evidence_k=4)

    assert _feature("number_coverage", fee) == 1.0
    assert _feature("number_coverage", growth) == 1.0
    assert _feature("number_coverage", no_percent) == 0.0


def test_no_signals_vector():
    vector = compute_features(CLAIM, [], evidence_k=4)

    expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.log1p(4), 0.0]
    np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-12)


def test_no_signals_claim_without_numbers_or_content_tokens():
    vector = compute_features("It is the", [], evidence_k=4)

    assert _feature("number_coverage", vector) == 1.0
    assert _feature("has_numbers", vector) == 0.0
    assert _feature("content_coverage", vector) == 1.0
    assert _feature("claim_len", vector) == math.log1p(3)


def test_none_rerank_ordered_last():
    tied_first = _signal("a", "x", NLIResult(0.3, 0.3, 0.4), 1.0, 0.1)
    unscored = _signal("b", "x", NLIResult(0.3, 0.3, 0.4), None, 0.1)
    tied_second = _signal("c", "x", NLIResult(0.3, 0.3, 0.4), 1.0, 0.1)
    lowest = _signal("d", "x", NLIResult(0.3, 0.3, 0.4), -2.0, 0.1)
    also_unscored = _signal("e", "x", NLIResult(0.3, 0.3, 0.4), None, 0.1)

    ordered = order_signals([unscored, lowest, tied_first, also_unscored, tied_second])

    # Rerank descending; equal scores and the unscored pair keep their input order.
    assert [s.passage_id for s in ordered] == ["a", "c", "d", "b", "e"]


def test_top_rerank_zero_when_no_signal_has_a_rerank_score():
    unscored = _signal("p1", "Refunds take 30 days.", NLIResult(0.9, 0.05, 0.05), None, 0.8)

    vector = compute_features(CLAIM, [unscored], evidence_k=4)

    assert _feature("top_rerank", vector) == 0.0
