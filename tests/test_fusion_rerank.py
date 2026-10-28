"""Reciprocal rank fusion and cross-encoder reranking of passage hits."""

import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest

from backend.retrieval.fusion import rrf_fuse
from backend.retrieval.rerank import rerank_hits
from backend.retrieval.types import PassageHit
from tests.fakes import KeywordReranker

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _hit(n: int, text: str, rank: int, scores: dict[str, float]) -> PassageHit:
    """A hit whose ids are derived from `n`, so each hit is distinguishable."""
    return PassageHit(
        passage_id=UUID(int=n),
        document_id=UUID(int=1000 + n),
        version_id=UUID(int=2000 + n),
        version_number=1,
        effective_from=_EPOCH,
        document_title=f"doc {n}",
        text=text,
        start_char=0,
        end_char=len(text),
        page_start=None,
        page_end=None,
        section=None,
        scores=scores,
        rank=rank,
    )


class _ExplodingReranker:
    """Any call fails the test: used to prove the model is not touched."""

    model_id = "fake-exploding"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise AssertionError("the reranker must not be called")


class _ThreadRecordingReranker:
    """Records which thread `score` ran on, and scores every passage 0."""

    model_id = "fake-thread-recorder"

    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.thread_ids.append(threading.get_ident())
        return [0.0 for _ in passages]


# --- rrf_fuse -------------------------------------------------------------------------------


def test_rrf_literals() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])

    assert [key for key, _ in fused] == ["b", "c", "a", "d"]
    expected = [0.03252247488101534, 0.03200204813108039, 0.01639344262295082, 0.015873015873015872]
    for (_, score), want in zip(fused, expected, strict=True):
        assert score == pytest.approx(want, abs=1e-15, rel=0)


def test_rrf_tie_break() -> None:
    # Full tie on score and on best rank: the key decides.
    assert [key for key, _ in rrf_fuse([["b"], ["a"]])] == ["a", "b"]

    # With k=1, m (ranks 1 and 5) and a (ranks 2 and 2) both score exactly 2/3, so the better
    # individual rank must put m first even though "a" sorts before "m".
    fused = rrf_fuse([["m", "a", "c", "d", "e"], ["x", "a", "y", "w", "m"]], k=1)
    scores = dict(fused)
    assert scores["m"] == scores["a"] == 0.6666666666666666
    # x scores 1/2; c and y tie at 1/4 with best rank 3, and d and w tie at 1/5 with best rank 4,
    # so each pair falls back to key order.
    assert [key for key, _ in fused] == ["m", "a", "x", "c", "y", "d", "w", "e"]


def test_rrf_duplicate_raises() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a", "b", "a"]])

    # The same key in two different rankings is the normal case, not an error.
    assert [key for key, _ in rrf_fuse([["a"], ["a"]])] == ["a"]


def test_rrf_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# --- rerank_hits ----------------------------------------------------------------------------


async def test_rerank_orders_and_renumbers() -> None:
    hits = [
        _hit(1, "the red car", rank=1, scores={"rrf": 0.5}),
        _hit(2, "a lazy dog", rank=2, scores={"rrf": 0.25}),
        _hit(3, "the red fox jumps high", rank=3, scores={"rrf": 0.125}),
    ]

    reranked = await rerank_hits(KeywordReranker(), "red fox jumps", hits, k=3)

    assert [hit.passage_id for hit in reranked] == [UUID(int=3), UUID(int=1), UUID(int=2)]
    assert [hit.rank for hit in reranked] == [1, 2, 3]
    assert [hit.scores["rerank"] for hit in reranked] == [3.0, 1.0, 0.0]
    # Earlier scores are kept alongside the new one.
    assert [hit.scores["rrf"] for hit in reranked] == [0.125, 0.5, 0.25]
    # The inputs are left exactly as they were: no shared or mutated scores dict.
    assert [hit.scores for hit in hits] == [{"rrf": 0.5}, {"rrf": 0.25}, {"rrf": 0.125}]
    assert [hit.rank for hit in hits] == [1, 2, 3]


async def test_rerank_truncates_to_k_and_returns_fewer_for_short_input() -> None:
    hits = [
        _hit(1, "the red car", rank=1, scores={}),
        _hit(2, "a lazy dog", rank=2, scores={}),
        _hit(3, "the red fox jumps high", rank=3, scores={}),
    ]

    top_two = await rerank_hits(KeywordReranker(), "red fox jumps", hits, k=2)
    assert [hit.passage_id for hit in top_two] == [UUID(int=3), UUID(int=1)]
    assert [hit.rank for hit in top_two] == [1, 2]

    everything = await rerank_hits(KeywordReranker(), "red fox jumps", hits, k=10)
    assert len(everything) == 3


async def test_rerank_equal_scores_keep_previous_rank_order() -> None:
    # Hits 2 and 3 both score 1.0. The input list is not in rank order, so only the previous
    # `rank` field (not list position) can put hit 2 ahead of hit 3.
    hit3 = _hit(3, "beta only", rank=3, scores={})
    hit1 = _hit(1, "nothing relevant", rank=1, scores={})
    hit2 = _hit(2, "alpha only", rank=2, scores={})

    reranked = await rerank_hits(KeywordReranker(), "alpha beta", [hit3, hit1, hit2], k=3)

    assert [hit.passage_id for hit in reranked] == [UUID(int=2), UUID(int=3), UUID(int=1)]
    assert [hit.scores["rerank"] for hit in reranked] == [1.0, 1.0, 0.0]


async def test_rerank_empty_no_call() -> None:
    assert await rerank_hits(_ExplodingReranker(), "anything", [], k=5) == []


async def test_rerank_scores_off_the_event_loop_thread() -> None:
    reranker = _ThreadRecordingReranker()
    hits = [_hit(1, "some text", rank=1, scores={})]

    await rerank_hits(reranker, "query", hits, k=1)

    assert len(reranker.thread_ids) == 1
    assert reranker.thread_ids[0] != threading.get_ident()
