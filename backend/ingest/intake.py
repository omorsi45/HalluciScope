"""Upload intake: validate, store the blob, and queue the ingestion job in one transaction.

Nothing here parses the file. The ``ingest_upload`` job (``backend.ingest.handlers``) does
that, so a slow or hostile file never occupies an API request.
"""

import hashlib
import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

from backend.api.errors import ApiError
from backend.auth.service import Principal
from backend.authz.policy import require_collection
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import audit, documents
from backend.ingest.errors import IngestError
from backend.ingest.parsers import mime_for
from backend.jobs.queue import enqueue
from backend.obs.logging import request_id_var

MAX_FILENAME_CHARS = 255
MAX_TITLE_CHARS = 300
# A client clock may run slightly ahead of the server's.
FUTURE_TOLERANCE = timedelta(seconds=60)
_PATH_SEPARATORS = re.compile(r"[\\/]")


@dataclass(frozen=True)
class IntakeResult:
    document_id: UUID
    upload_id: UUID
    status: Literal["queued", "duplicate"]


def _sanitize_filename(raw: str) -> str:
    """The basename without control characters, at most 255 characters, extension kept."""
    name = "".join(ch for ch in raw if unicodedata.category(ch) != "Cc")
    name = _PATH_SEPARATORS.split(name)[-1]
    if len(name) > MAX_FILENAME_CHARS:
        stem, extension = os.path.splitext(name)
        name = (stem[: max(0, MAX_FILENAME_CHARS - len(extension))] + extension)[
            :MAX_FILENAME_CHARS
        ]
    return name


def _document_not_found() -> ApiError:
    return ApiError(404, "document_not_found", "Document not found")


async def intake_upload(
    db: Database,
    principal: Principal,
    *,
    data: bytes,
    filename: str,
    collection_id: UUID | None = None,
    document_id: UUID | None = None,
    title: str | None = None,
    effective_from: datetime | None = None,
    settings: Settings,
    collection_kinds: Sequence[str] = ("normal",),
) -> IntakeResult:
    """Store the upload and queue its ingestion.

    Give ``collection_id`` for a new document or ``document_id`` for a new version of an
    existing one. An upload whose bytes equal the document's current version is recorded as
    ``duplicate`` and queues nothing.
    """
    if (collection_id is None) == (document_id is None):
        raise ValueError("give exactly one of collection_id and document_id")
    filename = _sanitize_filename(filename)
    try:
        mime_type = mime_for(filename)
    except IngestError:
        raise ApiError(415, "unsupported_type", "Unsupported file type") from None
    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "too_large", "File too large")
    now = datetime.now(UTC)
    if effective_from is not None and effective_from > now + FUTURE_TOLERANCE:
        raise ApiError(422, "effective_from_in_future", "effective_from is in the future")
    title = (title if title is not None else os.path.splitext(filename)[0])[:MAX_TITLE_CHARS]
    blob_sha256 = hashlib.sha256(data).digest()

    async with db.tenant_tx(principal.tenant_id) as conn:
        current_version_id: UUID | None = None
        current: dict | None = None
        if document_id is not None:
            document = await documents.get_document(conn, document_id)
            if document is None:
                raise _document_not_found()
            try:
                await require_collection(
                    conn, principal, document["collection_id"], "write", kinds=collection_kinds
                )
            except ApiError as exc:
                if exc.status == 404:
                    raise _document_not_found() from None
                raise
            current_version_id = document["current_version_id"]
            if current_version_id is not None:
                cur = await conn.execute(
                    "select blob_sha256, effective_from from document_versions where id = %s",
                    (current_version_id,),
                )
                current = await cur.fetchone()
        else:
            await require_collection(
                conn, principal, collection_id, "write", kinds=collection_kinds
            )
            document_id = await documents.insert_document(
                conn,
                tenant_id=principal.tenant_id,
                collection_id=collection_id,
                title=title,
                filename=filename,
                mime_type=mime_type,
                created_by=principal.user_id,
            )

        if effective_from is None:
            # Never earlier than the current version, or the worker would reject the new
            # version as out of order (the current one may be dated up to 60 s ahead, and the
            # wall clock can step backwards).
            effective_from = now if current is None else max(now, current["effective_from"])

        await documents.insert_blob(conn, principal.tenant_id, blob_sha256, data)
        status: Literal["queued", "duplicate"] = "queued"
        if current is not None and bytes(current["blob_sha256"]) == blob_sha256:
            status = "duplicate"
        cur = await conn.execute(
            "insert into uploads (tenant_id, document_id, blob_sha256, filename, declared_mime, "
            "effective_from, status, version_id, created_by) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s) returning id",
            (
                principal.tenant_id,
                document_id,
                blob_sha256,
                filename,
                mime_type,
                effective_from,
                status,
                current_version_id if status == "duplicate" else None,
                principal.user_id,
            ),
        )
        upload_id: UUID = (await cur.fetchone())["id"]
        if status == "queued":
            await enqueue(
                conn,
                "ingest_upload",
                {"upload_id": str(upload_id)},
                tenant_id=principal.tenant_id,
                priority=0,
                idempotency_key=f"upload:{upload_id}",
                request_id=request_id_var.get(),
            )
        await audit.record(
            conn,
            action="document_uploaded",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            target_type="document",
            target_id=str(document_id),
            details={"upload_id": str(upload_id)},
        )
    return IntakeResult(document_id, upload_id, status)
