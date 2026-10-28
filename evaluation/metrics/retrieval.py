"""Original retrieval metrics for the BEIR benchmark (no metric libraries).

Definitions follow trec_eval: nDCG uses linear gain (gain equals the relevance grade) with a
log2(rank + 1) discount, and a document is relevant only when its grade is greater than zero.
"""

import math
from collections.abc import Mapping, Sequence


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _check_unique(ranked: Sequence[str]) -> None:
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranking contains a repeated document")


def _relevant_grades(qrels: Mapping[str, int]) -> dict[str, int]:
    return {doc: grade for doc, grade in qrels.items() if grade > 0}


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """Normalized DCG at k with linear gain; raises ValueError when nothing is relevant."""
    _check_k(k)
    _check_unique(ranked)
    relevant = _relevant_grades(qrels)
    if not relevant:
        raise ValueError("query has no relevant documents")
    dcg = sum(
        relevant.get(doc, 0) / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Relevant documents in the top k over all relevant documents."""
    _check_k(k)
    _check_unique(ranked)
    relevant = _relevant_grades(qrels)
    if not relevant:
        raise ValueError("query has no relevant documents")
    return sum(1 for doc in ranked[:k] if doc in relevant) / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document in the top k, else 0."""
    _check_k(k)
    _check_unique(ranked)
    relevant = _relevant_grades(qrels)
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def maxp(passage_ranking: Sequence[str], passage_to_doc: Mapping[str, str]) -> list[str]:
    """Collapse a passage ranking to documents, ordered by each document's best passage.

    A document's MaxP score is its highest passage score, so for any score that ranks passages
    best first, the document order is the order of each document's first passage.
    """
    documents: dict[str, None] = {}
    for passage in passage_ranking:
        documents.setdefault(passage_to_doc[passage])
    return list(documents)


def evaluate_run(
    run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, float]:
    """Mean nDCG@10, Recall@10, Recall@100 and MRR@10 over queries with a relevant document.

    A judged query missing from the run counts as an empty ranking. Queries with no positive
    grade are skipped, and so are run queries absent from qrels. Any ranking that repeats a
    document raises ValueError.
    """
    for ranking in run.values():
        _check_unique(ranking)
    judged = {query: grades for query, grades in qrels.items() if _relevant_grades(grades)}
    if not judged:
        raise ValueError("no query has a relevant document")
    totals = {"ndcg@10": 0.0, "recall@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}
    for query, grades in judged.items():
        ranking = run.get(query, [])
        totals["ndcg@10"] += ndcg_at_k(ranking, grades, 10)
        totals["recall@10"] += recall_at_k(ranking, grades, 10)
        totals["recall@100"] += recall_at_k(ranking, grades, 100)
        totals["mrr@10"] += mrr_at_k(ranking, grades, 10)
    result = {name: total / len(judged) for name, total in totals.items()}
    result["n_queries"] = float(len(judged))
    return result
