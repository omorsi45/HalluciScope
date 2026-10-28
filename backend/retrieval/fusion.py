"""Reciprocal rank fusion over several best-first rankings of the same kind of key."""

from collections.abc import Hashable, Sequence
from typing import Any, TypeVar

# Keys must hash (to accumulate scores) and order (to break exact ties deterministically).
K = TypeVar("K", bound=Hashable)


def rrf_fuse(rankings: Sequence[Sequence[K]], k: int = 60) -> list[tuple[K, float]]:
    """Fuse best-first rankings into one list of ``(key, score)``, best first.

    A key at 1-based rank ``r`` in a ranking adds ``1 / (k + r)``; a key missing from a
    ranking adds nothing. Equal scores are ordered by the key's best (lowest) rank in any
    ranking, then by key ascending, so the result never depends on dict or set order. A key
    repeated inside one ranking is a caller bug and raises ``ValueError``.
    """
    scores: dict[K, float] = {}
    best_rank: dict[K, int] = {}
    for ranking in rankings:
        seen: set[K] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen:
                raise ValueError("a key appears more than once in one ranking")
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)

    def order(item: tuple[K, float]) -> tuple[float, int, Any]:
        key, score = item
        return (-score, best_rank[key], key)

    return sorted(scores.items(), key=order)
