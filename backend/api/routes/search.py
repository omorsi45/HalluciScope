"""POST /api/v1/search: BM25, dense, full-text or hybrid retrieval over the caller's collections.

Unreranked search runs in this process, which holds only the embedder (R7). Reranked search
needs the cross-encoder, which only the worker loads: the route stores the request, queues a
priority job and polls for its result with short transactions, never holding one across the wait.
"""

import asyncio
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from backend.api.errors import ApiError
from backend.api.schemas_v1 import PassageHitOut, SearchIn, SearchOut
from backend.auth.deps import current_principal, get_db
from backend.auth.service import Principal
from backend.authz.policy import readable_collection_ids
from backend.db.pool import Database
from backend.jobs.queue import enqueue
from backend.obs.logging import request_id_var
from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import SearchScope

router = APIRouter()

SEARCH_JOB_PRIORITY = 10
POLL_INTERVAL_S = 0.1


async def _database_now(conn: AsyncConnection) -> datetime:
    """The database clock, which stamps ``created_at`` and so is what ``known_at`` compares to."""
    cur = await conn.execute("select now() as now")
    return (await cur.fetchone())["now"]


async def _queue_request(
    db: Database,
    principal: Principal,
    body: SearchIn,
    scope: SearchScope,
) -> UUID:
    """Store the request and its job in one transaction; return the request id."""
    params: dict[str, Any] = {
        "k": body.k,
        "mode": body.mode,
        "collection_ids": [str(collection_id) for collection_id in scope.collection_ids],
        "as_of": None if scope.as_of is None else scope.as_of.isoformat(),
        "known_at": None if scope.known_at is None else scope.known_at.isoformat(),
    }
    async with db.tenant_tx(principal.tenant_id) as conn:
        cur = await conn.execute(
            "insert into search_requests (tenant_id, user_id, query, params) "
            "values (%s, %s, %s, %s) returning id",
            (principal.tenant_id, principal.user_id, body.query, Jsonb(params)),
        )
        request_id = (await cur.fetchone())["id"]
        await enqueue(
            conn,
            "search",
            {"search_request_id": str(request_id)},
            tenant_id=principal.tenant_id,
            priority=SEARCH_JOB_PRIORITY,
            request_id=request_id_var.get(),
        )
    return request_id


async def _read_request(
    db: Database, tenant_id: UUID, request_id: UUID, *, last_look: bool
) -> dict[str, Any] | None:
    """The request's status and hits, removing the row once it is settled or on a last look.

    Removing a row the worker has not started leaves its job a no-op, so a caller that gave up
    costs no rerank, and its query text does not linger.
    """
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select status, hits from search_requests where id = %s", (request_id,)
        )
        row = await cur.fetchone()
        if row is not None and (row["status"] != "queued" or last_look):
            await conn.execute("delete from search_requests where id = %s", (request_id,))
    return row


async def _wait_for_hits(
    db: Database, tenant_id: UUID, request_id: UUID, wait_s: float
) -> list[PassageHitOut]:
    deadline = time.monotonic() + wait_s
    while True:
        last_look = time.monotonic() >= deadline
        row = await _read_request(db, tenant_id, request_id, last_look=last_look)
        if row is None or row["status"] == "failed":
            raise ApiError(502, "search_failed", "Search failed")
        if row["status"] == "done":
            return [PassageHitOut.model_validate(hit) for hit in row["hits"]]
        if last_look:
            raise ApiError(504, "search_timeout", "Search timed out")
        await asyncio.sleep(min(POLL_INTERVAL_S, max(deadline - time.monotonic(), 0.0)))


@router.post("/search", response_model=SearchOut)
async def search(
    request: Request,
    body: SearchIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> SearchOut:
    started = time.perf_counter()
    async with db.tenant_tx(principal.tenant_id) as conn:
        collection_ids = await readable_collection_ids(conn, principal, body.collection_ids)
        # An as-of query sees what was known when it was asked: pin that moment now, so a
        # queued rerank is not affected by how long the job waits.
        known_at = await _database_now(conn) if body.as_of is not None else None
    scope = SearchScope(principal.tenant_id, collection_ids, as_of=body.as_of, known_at=known_at)

    if body.rerank:
        request_id = await _queue_request(db, principal, body, scope)
        hits = await _wait_for_hits(
            db, principal.tenant_id, request_id, request.app.state.settings.search_wait_s
        )
    else:
        retriever = PostgresRetriever(db, request.app.state.embedder, None)
        found = await retriever.search(body.query, scope, k=body.k, mode=body.mode, rerank=False)
        hits = [PassageHitOut.model_validate(hit) for hit in found]
    return SearchOut(
        hits=hits,
        mode=body.mode,
        reranked=body.rerank,
        took_ms=int((time.perf_counter() - started) * 1000),
    )
