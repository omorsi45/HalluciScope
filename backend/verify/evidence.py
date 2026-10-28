"""Per-claim evidence gathering, sentence-window spans, and the evidence relation rule.

Offsets are Python `str` indices into the version text: a passage hit starts at
`hit.start_char`, so `version_text[start:end]` is exactly the span a window reports.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal
from uuid import UUID

from backend.ingest.segment import segment_sentences
from backend.models.protocols import Reranker
from backend.retrieval.rerank import rerank_hits
from backend.retrieval.types import PassageHit, Retriever, SearchScope
from backend.verify.types import Claim, NLIResult


async def gather_evidence(
    retriever: Retriever,
    reranker: Reranker,
    claims: Sequence[Claim],
    scope: SearchScope,
    *,
    pinned: Mapping[int, Sequence[PassageHit]] = {},
    depth: int = 8,
    candidates: int = 20,
    concurrency: int = 4,
) -> list[list[PassageHit]]:
    """The evidence passages of each claim, in claim order; at most `concurrency` run at once.

    Each claim retrieves `candidates` passages (hybrid, no rerank), adds the passages pinned to
    its ordinal (the ones its answer cited), and reranks the union against the claim text. Every
    pinned passage is kept, then the best reranked others fill the list up to `depth`. A claim
    with nothing retrieved and nothing pinned gets an empty list.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def gather_one(claim: Claim) -> list[PassageHit]:
        async with semaphore:
            found = await retriever.search(
                claim.text,
                scope,
                k=candidates,
                mode="hybrid",
                rerank=False,
                candidates=candidates,
            )
            return await _select_evidence(
                reranker, claim, pinned.get(claim.ordinal, ()), found, depth
            )

    return list(await asyncio.gather(*(gather_one(claim) for claim in claims)))


async def _select_evidence(
    reranker: Reranker,
    claim: Claim,
    pinned: Sequence[PassageHit],
    found: Sequence[PassageHit],
    depth: int,
) -> list[PassageHit]:
    union: dict[UUID, PassageHit] = {}
    for hit in (*pinned, *found):
        union.setdefault(hit.passage_id, hit)
    # Number the union in order, pinned first, so rerank ties keep that order.
    ordered = [replace(hit, rank=position) for position, hit in enumerate(union.values(), 1)]
    ranked = await rerank_hits(reranker, claim.text, ordered, k=len(ordered))

    pinned_ids = {hit.passage_id for hit in pinned}
    others = [hit for hit in ranked if hit.passage_id not in pinned_ids]
    keep = pinned_ids | {hit.passage_id for hit in others[: max(depth - len(pinned_ids), 0)]}
    selected = [hit for hit in ranked if hit.passage_id in keep]
    return [replace(hit, rank=position) for position, hit in enumerate(selected, 1)]


def window_spans(hit: PassageHit, widths: Sequence[int] = (1, 2)) -> list[tuple[int, int, str]]:
    """Runs of consecutive sentences of the passage as `(start, end, text)`, offsets absolute.

    Windows come width by width, each width left to right. A width larger than the sentence
    count gives none. `text` equals `hit.text[start - hit.start_char : end - hit.start_char]`.
    A passage without any sentence gives the whole passage as its one window.
    """
    sentences = segment_sentences(hit.text)
    if not sentences:
        return [(hit.start_char, hit.start_char + len(hit.text), hit.text)]
    windows = []
    for width in widths:
        for first in range(len(sentences) - width + 1):
            start = sentences[first].start
            end = sentences[first + width - 1].end
            windows.append((hit.start_char + start, hit.start_char + end, hit.text[start:end]))
    return windows


def relation_for(r: NLIResult) -> Literal["supports", "contradicts", "related"]:
    """`supports` when entail is the largest probability and above contra, `contradicts` when
    contra is the largest, else `related`. Ties for the largest count as the largest."""
    if r.entail >= r.neutral and r.entail > r.contra:
        return "supports"
    if r.contra >= r.entail and r.contra >= r.neutral:
        return "contradicts"
    return "related"
