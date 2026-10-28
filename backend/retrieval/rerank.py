"""Cross-encoder reranking of an already retrieved candidate list."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from backend.models.protocols import Reranker
from backend.retrieval.types import PassageHit


async def rerank_hits(
    reranker: Reranker, query: str, hits: Sequence[PassageHit], k: int
) -> list[PassageHit]:
    """Score ``hits`` against ``query`` and return the best ``k`` as new hits, best first.

    Each returned hit keeps its earlier scores and gains ``scores["rerank"]`` (the raw logit);
    ``rank`` is renumbered from 1. Equal logits keep their previous relative order. The result
    is shorter than ``k`` when fewer than ``k`` hits come in, and empty input never reaches the
    model. Scoring is CPU-bound, so it runs on a worker thread to keep the event loop free.
    """
    if not hits:
        return []
    logits = await asyncio.to_thread(reranker.score, query, [hit.text for hit in hits])
    # strict: a model that returns the wrong number of scores must fail, not drop hits silently.
    scored = sorted(zip(hits, logits, strict=True), key=lambda pair: (-pair[1], pair[0].rank))
    return [
        replace(hit, scores={**hit.scores, "rerank": logit}, rank=position)
        for position, (hit, logit) in enumerate(scored[:k], start=1)
    ]
