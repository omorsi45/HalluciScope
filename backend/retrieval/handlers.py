"""The search job: reranked search for a request the API is waiting on, and its failure hook.

The query text lives only in the RLS table ``search_requests``; the job payload is the request
id. Failures are reduced to the stable code ``search_failed``: exception messages and query text
never reach a log line or a row.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from backend.api.schemas_v1 import PassageHitOut
from backend.db.pool import Database
from backend.jobs.worker import JobContext, PermanentJobError
from backend.obs.logging import bind
from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import SearchScope

logger = logging.getLogger(__name__)

SEARCH_FAILED = "search_failed"


async def _set_failed(db: Database, tenant_id: UUID, request_id: UUID, error_code: str) -> None:
    """Fail the request unless it already reached a final status (done or failed)."""
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute(
            "update search_requests set status = 'failed', error_code = %s, completed_at = now() "
            "where id = %s and status = 'queued'",
            (error_code, request_id),
        )


def _timestamp(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


async def _search(services: Any, tenant_id: UUID, request: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the stored request with the reranker; return the hits as JSON-ready dicts."""
    params = request["params"]
    scope = SearchScope(
        tenant_id,
        tuple(UUID(collection_id) for collection_id in params["collection_ids"]),
        as_of=_timestamp(params["as_of"]),
        known_at=_timestamp(params["known_at"]),
    )
    retriever = PostgresRetriever(services.db, services.embedder, services.reranker)
    hits = await retriever.search(
        request["query"], scope, k=params["k"], mode=params["mode"], rerank=True
    )
    return [PassageHitOut.model_validate(hit).model_dump(mode="json") for hit in hits]


async def handle_search(ctx: JobContext) -> None:
    tenant_id = ctx.job.tenant_id
    request_id = UUID(ctx.job.payload["search_request_id"])
    with bind(tenant_id=str(tenant_id)):
        async with ctx.db.tenant_tx(tenant_id) as conn:
            cur = await conn.execute(
                "select query, params, status from search_requests where id = %s", (request_id,)
            )
            request = await cur.fetchone()
        # A missing row means the caller timed out and removed it; a settled one needs no rerun.
        if request is None or request["status"] != "queued":
            return

        try:
            hits = await _search(ctx.services, tenant_id, request)
        except Exception:
            logger.exception("search job error")
            await _set_failed(ctx.db, tenant_id, request_id, SEARCH_FAILED)
            raise PermanentJobError(SEARCH_FAILED) from None

        async with ctx.db.tenant_tx(tenant_id) as conn:
            await conn.execute(
                "update search_requests set status = 'done', hits = %s, completed_at = now() "
                "where id = %s and status = 'queued'",
                (Jsonb(hits), request_id),
            )
            # Rows nobody collected (a caller that disconnected) must not pile up. Row level
            # security confines the delete to this tenant.
            await conn.execute(
                "delete from search_requests where created_at < now() - interval '1 hour'"
            )
        logger.info("search finished", extra={"outcome": "done", "n_passages": len(hits)})


async def fail_search(ctx: JobContext, error_code: str) -> None:
    """Failure hook: a search job that died (retries used up, lease expired) fails its request.

    Idempotent, and never raises: the worker must be able to settle other jobs after it.
    """
    try:
        await _set_failed(
            ctx.db, ctx.job.tenant_id, UUID(ctx.job.payload["search_request_id"]), error_code
        )
    except Exception:
        logger.exception("search failure hook error")
