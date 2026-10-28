"""Contracts shared by every retrieval mode."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

SearchMode = Literal["hybrid", "bm25", "dense", "fts"]


class RerankUnavailable(RuntimeError):
    """The reranker is not configured or failed to load."""


@dataclass(frozen=True)
class SearchScope:
    """What a caller may see.

    ``collection_ids`` must already be filtered to collections the principal can read.
    ``as_of`` is valid time and ``known_at`` is transaction time (R9); both ``None`` means the
    current versions.
    """

    tenant_id: UUID
    collection_ids: tuple[UUID, ...]
    as_of: datetime | None = None
    known_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        return self.as_of is None and self.known_at is None


@dataclass(frozen=True)
class PassageHit:
    passage_id: UUID
    document_id: UUID
    version_id: UUID
    version_number: int
    effective_from: datetime
    document_title: str
    text: str
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    section: str | None
    scores: dict[str, float]  # keys among 'bm25', 'dense', 'fts', 'rrf', 'rerank'
    rank: int  # 1-based in the returned list


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        scope: SearchScope,
        *,
        k: int = 10,
        mode: SearchMode = "hybrid",
        rerank: bool = True,
        candidates: int = 50,
        exact: bool = False,
    ) -> list[PassageHit]: ...
