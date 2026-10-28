"""PostgresRetriever: BM25, dense, full-text, or their RRF fusion, with optional rerank.

Every query runs inside one tenant transaction over the caller's visible passage set (access
control and bitemporal scope, see ``visible.py``). The query embedding is computed before the
transaction opens so that no pooled connection waits on the model, and reranking runs after it
closes for the same reason.
"""

import asyncio
import time
from typing import get_args
from uuid import UUID

import numpy as np
from psycopg import AsyncConnection

from backend.db.pool import Database
from backend.models.protocols import Embedder, Reranker
from backend.obs.metrics import STAGE_SECONDS
from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.fts import fts_search
from backend.retrieval.fusion import rrf_fuse
from backend.retrieval.rerank import rerank_hits
from backend.retrieval.types import PassageHit, RerankUnavailable, SearchMode, SearchScope
from backend.retrieval.visible import visible_version_ids

MAX_K = 300
MAX_CANDIDATES = 1000

# One row per passage id: the passage, its version's number and valid-time start, and the title.
_HYDRATE_SQL = """
select p.id, p.document_id, p.version_id, v.version_number, v.effective_from,
    d.title as document_title, p.text, p.start_char, p.end_char,
    p.page_start, p.page_end, p.section
from passages p
join document_versions v on v.id = p.version_id
join documents d on d.id = p.document_id
where p.id = any(%s::uuid[])
"""


class PostgresRetriever:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        k1: float = 0.9,
        b: float = 0.4,
        rrf_k: int = 60,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._reranker = reranker
        self._k1 = k1
        self._b = b
        self._rrf_k = rrf_k

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
    ) -> list[PassageHit]:
        """The top ``k`` visible passages for ``query``, best first, ranks ``1..k``.

        ``candidates`` is how many passages each component fetches in hybrid mode, and how many
        the reranker scores; the fetch is never smaller than ``k``. ``exact`` makes dense search
        scan the visible passages instead of using the HNSW index.
        """
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}")
        if not 1 <= candidates <= MAX_CANDIDATES:
            raise ValueError(f"candidates must be between 1 and {MAX_CANDIDATES}")
        if mode not in get_args(SearchMode):
            raise ValueError(f"mode must be one of {get_args(SearchMode)}")
        reranker = self._reranker if rerank else None
        if rerank and reranker is None:
            raise RerankUnavailable("no reranker is configured")
        if not scope.collection_ids:
            return []

        started = time.perf_counter()
        try:
            query_vec = None
            if mode in ("hybrid", "dense"):
                vectors = await asyncio.to_thread(self._embedder.embed_queries, [query])
                query_vec = vectors[0]
            # The component fetch: deep enough for the fusion or the reranker, else just k.
            n = max(k, candidates) if mode == "hybrid" or rerank else k
            async with self._db.tenant_tx(scope.tenant_id) as conn:
                version_ids = await visible_version_ids(conn, scope)
                scores = await self._component_scores(
                    conn, query, query_vec, scope, mode, n, version_ids, exact
                )
                # Only what survives to the end needs its text: the reranker's pool, or k.
                hits = await _hydrate(conn, scores, n if rerank else k)
            if reranker is None:
                return hits
            return await rerank_hits(reranker, query, hits, k)
        finally:
            STAGE_SECONDS.labels(stage="retrieve").observe(time.perf_counter() - started)

    async def _component_scores(
        self,
        conn: AsyncConnection,
        query: str,
        query_vec: np.ndarray | None,
        scope: SearchScope,
        mode: SearchMode,
        n: int,
        version_ids: list[UUID] | None,
        exact: bool,
    ) -> dict[UUID, dict[str, float]]:
        """Scores per passage id, best first: the component scores, plus ``rrf`` when fused.

        Lexical statements run before dense search: its ``SET LOCAL`` settings (and, in exact
        mode, disabled index scans) stay in effect for the rest of the transaction.
        """
        rankings: dict[str, list[tuple[UUID, float]]] = {}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = await bm25_search(
                conn, query, scope, limit=n, version_ids=version_ids, k1=self._k1, b=self._b
            )
        if mode == "fts":
            rankings["fts"] = await fts_search(conn, query, scope, limit=n, version_ids=version_ids)
        if mode in ("dense", "hybrid"):
            rankings["dense"] = await dense_search(
                conn, query_vec, scope, limit=n, version_ids=version_ids, exact=exact
            )

        components: dict[UUID, dict[str, float]] = {}
        for name, ranking in rankings.items():
            for passage_id, score in ranking:
                components.setdefault(passage_id, {})[name] = float(score)
        if mode != "hybrid":
            (only,) = rankings.values()
            return {passage_id: components[passage_id] for passage_id, _ in only}
        fused = rrf_fuse(
            [[passage_id for passage_id, _ in ranking] for ranking in rankings.values()],
            k=self._rrf_k,
        )
        return {passage_id: {**components[passage_id], "rrf": rrf} for passage_id, rrf in fused}


async def _hydrate(
    conn: AsyncConnection, scores: dict[UUID, dict[str, float]], keep: int
) -> list[PassageHit]:
    """The first ``keep`` passages of ``scores`` as hits, in that order, ranked from 1."""
    ids = list(scores)[:keep]
    if not ids:
        return []
    cur = await conn.execute(_HYDRATE_SQL, (ids,))
    rows = {row["id"]: row for row in await cur.fetchall()}
    hits = []
    for rank, passage_id in enumerate(ids, start=1):
        row = rows[passage_id]
        hits.append(
            PassageHit(
                passage_id=passage_id,
                document_id=row["document_id"],
                version_id=row["version_id"],
                version_number=row["version_number"],
                effective_from=row["effective_from"],
                document_title=row["document_title"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section=row["section"],
                scores=scores[passage_id],
                rank=rank,
            )
        )
    return hits
