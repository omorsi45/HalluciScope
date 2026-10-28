"""Retrieval metrics: every expected value below is derived by hand, not by the code under test."""

import pytest

from evaluation.metrics.retrieval import (
    evaluate_run,
    maxp,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

TOL = 1e-12


def test_ndcg_literal():
    # DCG = 1/log2(2) + 2/log2(4) = 2; IDCG = 2/log2(2) + 1/log2(3); ratio checked by hand.
    value = ndcg_at_k(["d1", "d2", "d3"], {"d1": 1, "d3": 2}, 3)
    assert value == pytest.approx(0.7601875334318685, abs=TOL)


def test_ndcg_perfect_and_zero():
    qrels = {"d1": 1, "d3": 2}
    assert ndcg_at_k(["d3", "d1", "d2"], qrels, 3) == pytest.approx(1.0, abs=TOL)
    assert ndcg_at_k(["x", "y", "z"], qrels, 3) == 0.0


def test_ndcg_ignores_nonpositive_grades():
    # Only d1 is relevant and sits at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.
    # A grade of -1 or 0 must neither count as gain nor enter the ideal ranking.
    qrels = {"d1": 1, "d2": 0, "d3": -1}
    assert ndcg_at_k(["d2", "d3", "d1"], qrels, 3) == pytest.approx(0.5, abs=TOL)


def test_ndcg_ideal_uses_only_k_best_grades():
    # k=1: the ideal list is [3], so retrieving the grade-2 document first scores 2/3.
    qrels = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["b"], qrels, 1) == pytest.approx(2 / 3, abs=TOL)


def test_ndcg_counts_only_the_top_k():
    # The relevant document at rank 2 lies outside k=1.
    assert ndcg_at_k(["x", "a"], {"a": 1}, 1) == 0.0


def test_ndcg_no_relevant_documents_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 0, "b": -2}, 10)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {}, 10)


def test_recall_and_mrr():
    ranked = ["a", "b", "c"]
    qrels = {"b": 1, "z": 1}
    assert recall_at_k(ranked, qrels, 3) == 0.5
    assert mrr_at_k(ranked, qrels, 3) == 0.5


def test_recall_with_empty_ranking_is_zero():
    assert recall_at_k([], {"a": 1}, 10) == 0.0


def test_recall_counts_only_positive_grades_and_the_top_k():
    qrels = {"a": 1, "b": 0, "c": 2}
    # b has grade 0, so the relevant set is {a, c}; only c is inside the top 2.
    assert recall_at_k(["b", "c", "a"], qrels, 2) == 0.5


def test_recall_without_relevant_documents_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 0}, 10)


def test_mrr_uses_first_relevant_and_respects_k():
    qrels = {"c": 1, "d": 1}
    ranked = ["a", "b", "c", "d"]
    assert mrr_at_k(ranked, qrels, 4) == pytest.approx(1 / 3, abs=TOL)
    assert mrr_at_k(ranked, qrels, 2) == 0.0


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k, mrr_at_k])
@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(metric, k):
    with pytest.raises(ValueError):
        metric(["a"], {"a": 1}, k)


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k, mrr_at_k])
def test_duplicate_docs_raise_in_each_metric(metric):
    with pytest.raises(ValueError):
        metric(["a", "b", "a"], {"a": 1}, 3)


def test_maxp_first_occurrence():
    passage_to_doc = {"p1": "A", "p2": "B", "p3": "A"}
    assert maxp(["p1", "p2", "p3"], passage_to_doc) == ["A", "B"]
    assert maxp(["p3", "p2", "p1"], passage_to_doc) == ["A", "B"]
    assert maxp(["p2", "p1", "p3"], passage_to_doc) == ["B", "A"]
    assert maxp([], passage_to_doc) == []


def test_maxp_unknown_passage_raises():
    with pytest.raises(KeyError):
        maxp(["p1", "missing"], {"p1": "A"})


def test_evaluate_run_returns_the_documented_keys():
    result = evaluate_run({"q1": ["a"]}, {"q1": {"a": 1}})
    assert set(result) == {"ndcg@10", "recall@10", "recall@100", "mrr@10", "n_queries"}
    assert result["n_queries"] == 1.0


def test_evaluate_run_averages_over_queries():
    run = {"q1": ["a", "b"], "q2": ["x"]}
    qrels = {"q1": {"b": 1}, "q2": {"x": 1}}
    result = evaluate_run(run, qrels)
    # q1: ndcg = (1/log2(3)) / 1 = 0.6309297535714574, mrr = 0.5, recall = 1. q2: all 1.
    assert result["ndcg@10"] == pytest.approx((0.6309297535714574 + 1.0) / 2, abs=TOL)
    assert result["mrr@10"] == pytest.approx(0.75, abs=TOL)
    assert result["recall@10"] == pytest.approx(1.0, abs=TOL)
    assert result["recall@100"] == pytest.approx(1.0, abs=TOL)
    assert result["n_queries"] == 2.0


def test_evaluate_run_missing_query_counts_zero():
    run = {"q1": ["a"]}
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    result = evaluate_run(run, qrels)
    assert result["ndcg@10"] == pytest.approx(0.5, abs=TOL)
    assert result["recall@10"] == pytest.approx(0.5, abs=TOL)
    assert result["recall@100"] == pytest.approx(0.5, abs=TOL)
    assert result["mrr@10"] == pytest.approx(0.5, abs=TOL)
    assert result["n_queries"] == 2.0


def test_evaluate_run_recall_100_uses_depth_100():
    # "rel" is at rank 100 and "late" at rank 101: recall@100 sees only the first of the two.
    ranking = [f"x{i}" for i in range(99)] + ["rel", "late"]
    result = evaluate_run({"q1": ranking}, {"q1": {"rel": 1, "late": 1}})
    assert result["recall@100"] == pytest.approx(0.5, abs=TOL)
    assert result["recall@10"] == 0.0
    assert result["ndcg@10"] == 0.0
    assert result["mrr@10"] == 0.0


def test_queries_without_relevant_excluded():
    run = {"q1": ["a"], "q2": ["x"], "q3": ["y"]}
    qrels = {"q1": {"a": 1}, "q2": {"x": 0}, "q3": {}}
    result = evaluate_run(run, qrels)
    assert result["n_queries"] == 1.0
    assert result["ndcg@10"] == pytest.approx(1.0, abs=TOL)


def test_evaluate_run_ignores_run_queries_absent_from_qrels():
    result = evaluate_run({"q1": ["a"], "extra": ["a"]}, {"q1": {"a": 1}})
    assert result["n_queries"] == 1.0


def test_evaluate_run_without_judged_queries_raises():
    with pytest.raises(ValueError):
        evaluate_run({"q1": ["a"]}, {"q1": {"a": 0}})
    with pytest.raises(ValueError):
        evaluate_run({}, {})


def test_duplicate_docs_raise():
    with pytest.raises(ValueError):
        evaluate_run({"q1": ["a", "b", "a"]}, {"q1": {"a": 1}})


def test_duplicate_docs_raise_even_for_unjudged_queries():
    # A repeated document means the run skipped MaxP collapse, whichever query it is in.
    with pytest.raises(ValueError):
        evaluate_run({"q1": ["a"], "q2": ["b", "b"]}, {"q1": {"a": 1}})
