"""The ingest_upload job: parse a stored upload, append a version, and settle the upload row.

``process_upload`` does the work and is idempotent. The job runs it for an upload queued by
``intake_upload``; a quick analysis runs it inline for its own upload.

Failures are reduced to stable codes. Exception messages, filenames and text never reach a
log line or a job row.
"""

import logging
from collections.abc import Sequence
from typing import Any, NoReturn
from uuid import UUID

import numpy as np
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from backend.db.pool import Database
from backend.db.repos import documents
from backend.ingest.errors import IngestError
from backend.ingest.parsers import parse_with_timeout
from backend.ingest.service import VersionResult, create_version
from backend.ingest.types import ParsedDocument
from backend.jobs.queue import enqueue
from backend.jobs.worker import JobContext, PermanentJobError, RetryableJobError
from backend.models.protocols import Embedder
from backend.obs.logging import bind, request_id_var
from backend.services import Services

logger = logging.getLogger(__name__)

# Statuses an upload can still leave; created_version, duplicate and failed are final.
_OPEN = ("queued", "processing")
_UPLOAD_SQL = (
    "select id, document_id, blob_sha256, filename, effective_from, status, created_by "
    "from uploads where id = %s"
)


class _EmbedFailed(Exception):
    """Stands in for whatever the embedding model raised."""


class _EmbeddingGuard:
    """Wraps the embedder so a model failure can be told apart from a database failure."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.dim = inner.dim

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        try:
            return self._inner.embed_passages(texts)
        except Exception:
            raise _EmbedFailed from None

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.embed_queries(texts)


async def _load_upload(
    conn: AsyncConnection, upload_id: UUID, *, lock: bool = False
) -> dict[str, Any] | None:
    cur = await conn.execute(_UPLOAD_SQL + (" for update" if lock else ""), (upload_id,))
    return await cur.fetchone()


async def _set_failed(db: Database, tenant_id: UUID, upload_id: UUID, error_code: str) -> None:
    """Fail the upload unless it already reached a final status."""
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute(
            "update uploads set status = 'failed', error_code = %s, updated_at = now() "
            "where id = %s and status = any(%s)",
            (error_code, upload_id, list(_OPEN)),
        )


async def _fail_permanently(db: Database, tenant_id: UUID, upload_id: UUID, code: str) -> NoReturn:
    await _set_failed(db, tenant_id, upload_id, code)
    raise PermanentJobError(code)


async def _append_version(
    services: Services, tenant_id: UUID, upload: dict[str, Any], parsed: ParsedDocument
) -> VersionResult | None:
    """Create the version, settle the upload and queue re-verification in one transaction.

    Returns None when the upload reached a final status in the meantime.
    """
    async with services.db.tenant_tx(tenant_id) as conn:
        # The row lock makes a second worker that ended up on this upload wait, then see it final.
        current = await _load_upload(conn, upload["id"], lock=True)
        if current["status"] not in _OPEN:
            return None
        result = await create_version(
            conn,
            _EmbeddingGuard(services.embedder),
            tenant_id=tenant_id,
            document_id=upload["document_id"],
            parsed=parsed,
            blob_sha256=bytes(upload["blob_sha256"]),
            effective_from=upload["effective_from"],
            created_by=upload["created_by"],
            settings=services.settings,
        )
        await conn.execute(
            "update uploads set status = %s, version_id = %s, stats = %s, updated_at = now() "
            "where id = %s",
            (
                result.status,
                result.version_id,
                Jsonb(
                    {
                        "n_passages": result.n_passages,
                        "n_reused": result.n_reused,
                        "n_new": result.n_new,
                    }
                ),
                upload["id"],
            ),
        )
        if result.status == "created_version" and result.previous_version_id is not None:
            await enqueue(
                conn,
                "reverify_document",
                {
                    "document_id": str(upload["document_id"]),
                    "old_version_id": str(result.previous_version_id),
                    "new_version_id": str(result.version_id),
                },
                tenant_id=tenant_id,
                priority=5,
                idempotency_key=f"reverify:{result.version_id}",
                request_id=request_id_var.get(),
            )
    logger.info(
        "upload ingested",
        extra={
            "outcome": result.status,
            "n_passages": result.n_passages,
            "n_reused": result.n_reused,
            "n_new": result.n_new,
            "version_number": result.version_number,
        },
    )
    return result


async def process_upload(
    services: Services, tenant_id: UUID, upload_id: UUID
) -> VersionResult | None:
    """Parse the stored upload and append its version; the work of the ``ingest_upload`` job.

    Idempotent: returns None, doing nothing, for an upload that is missing or already final.
    Otherwise returns the created version, or the current one when the content is unchanged.
    An ingestion failure fails the upload row and raises ``PermanentJobError(code)``; a failed
    embedding raises ``RetryableJobError("embed_failed")`` and leaves the upload open.
    """
    with bind(tenant_id=str(tenant_id)):
        async with services.db.tenant_tx(tenant_id) as conn:
            upload = await _load_upload(conn, upload_id)
            if upload is None or upload["status"] not in _OPEN:
                return None
            await conn.execute(
                "update uploads set status = 'processing', updated_at = now() where id = %s",
                (upload_id,),
            )
            data = await documents.get_blob(conn, bytes(upload["blob_sha256"]))

        # Outside any transaction: parsing can take up to the parse timeout.
        try:
            parsed = await parse_with_timeout(
                data, upload["filename"], services.settings.parse_timeout_s
            )
        except IngestError as exc:
            await _fail_permanently(services.db, tenant_id, upload_id, exc.code)

        try:
            return await _append_version(services, tenant_id, upload, parsed)
        except IngestError as exc:
            await _fail_permanently(services.db, tenant_id, upload_id, exc.code)
        except LookupError as exc:
            if type(exc) is not LookupError:
                # KeyError and IndexError are bugs, not a missing document.
                raise
            await _fail_permanently(services.db, tenant_id, upload_id, "document_deleted")
        except _EmbedFailed:
            raise RetryableJobError("embed_failed") from None


async def handle_ingest_upload(ctx: JobContext) -> None:
    await process_upload(ctx.services, ctx.job.tenant_id, UUID(ctx.job.payload["upload_id"]))


async def fail_ingest_upload(ctx: JobContext, error_code: str) -> None:
    """Failure hook: a job that died (retries used up, lease expired) fails its upload row.

    Idempotent, and never raises: the worker must be able to settle other jobs after it.
    """
    try:
        await _set_failed(
            ctx.db, ctx.job.tenant_id, UUID(ctx.job.payload["upload_id"]), error_code
        )
    except Exception:
        logger.exception("ingest upload failure hook error")
