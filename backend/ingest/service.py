"""Version creation: content dedupe, incremental embeddings, and passage insertion.

``create_version`` runs inside a transaction owned by the caller (a ``tenant_tx`` connection),
so any failure rolls the whole version back.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

import numpy as np
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from backend.config import Settings
from backend.ingest.errors import IngestError
from backend.ingest.passages import build_passages
from backend.ingest.types import ParsedDocument
from backend.models.protocols import Embedder
from backend.obs.metrics import INGEST_PASSAGES

# Read at call time, so tests can lower it with monkeypatch.
MAX_PASSAGES = 5_000


@dataclass(frozen=True)
class VersionResult:
    status: Literal["created_version", "duplicate"]
    version_id: UUID
    version_number: int
    previous_version_id: UUID | None
    n_passages: int
    n_reused: int
    n_new: int


async def create_version(
    conn: AsyncConnection,
    embedder: Embedder,
    *,
    tenant_id: UUID,
    document_id: UUID,
    parsed: ParsedDocument,
    blob_sha256: bytes,
    effective_from: datetime,
    created_by: UUID | None,
    settings: Settings,
) -> VersionResult:
    """Append a version of the document, or report a duplicate of the current one.

    Raises ``LookupError`` when the document is missing or soft-deleted, and ``IngestError``
    with ``effective_from_out_of_order``, ``too_many_passages``, or ``no_text``.
    """
    cur = await conn.execute(
        "select id, collection_id, current_version_id, deleted_at from documents "
        "where id = %s for update",
        (document_id,),
    )
    document = await cur.fetchone()
    if document is None or document["deleted_at"] is not None:
        raise LookupError("document not found")

    content_sha256 = hashlib.sha256(parsed.text.encode()).digest()
    previous_id: UUID | None = document["current_version_id"]
    if previous_id is not None:
        cur = await conn.execute(
            "select version_number, content_sha256, effective_from from document_versions "
            "where id = %s",
            (previous_id,),
        )
        previous = await cur.fetchone()
        # R5: only the current version is compared, so a revert is a new version.
        if bytes(previous["content_sha256"]) == content_sha256:
            return VersionResult(
                "duplicate", previous_id, previous["version_number"], None, 0, 0, 0
            )
        if effective_from < previous["effective_from"]:
            raise IngestError("effective_from_out_of_order")

    drafts = build_passages(parsed.text, parsed.pages, parsed.sections)
    if len(drafts) > MAX_PASSAGES:
        raise IngestError("too_many_passages")
    if not drafts:
        raise IngestError("no_text")

    embeddings = await _reusable_embeddings(
        conn, [draft.text_sha256 for draft in drafts], embedder.model_id
    )
    missing: dict[bytes, str] = {}
    for draft in drafts:
        if draft.text_sha256 not in embeddings:
            missing.setdefault(draft.text_sha256, draft.text)
    hashes = list(missing)
    for first in range(0, len(hashes), settings.embed_batch_size):
        batch = hashes[first : first + settings.embed_batch_size]
        vectors = await asyncio.to_thread(embedder.embed_passages, [missing[sha] for sha in batch])
        for sha, vector in zip(batch, vectors, strict=True):
            embeddings[sha] = np.asarray(vector, dtype=np.float32)

    cur = await conn.execute(
        "insert into document_versions (tenant_id, document_id, version_number, effective_from, "
        "content_sha256, blob_sha256, text, page_map, sections, metadata, parser, "
        "parser_version, char_count, created_by) "
        "values (%s, %s, (select coalesce(max(version_number), 0) + 1 from document_versions "
        "where document_id = %s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "returning id, version_number",
        (
            tenant_id,
            document_id,
            document_id,
            effective_from,
            content_sha256,
            blob_sha256,
            parsed.text,
            Jsonb([[page.page, page.start, page.end] for page in parsed.pages]),
            Jsonb([[span.start, span.end, span.path] for span in parsed.sections]),
            Jsonb(parsed.metadata),
            parsed.parser,
            parsed.parser_version,
            len(parsed.text),
            created_by,
        ),
    )
    version = await cur.fetchone()

    async with conn.cursor() as insert:
        await insert.executemany(
            "insert into passages (tenant_id, document_id, version_id, collection_id, ordinal, "
            "start_char, end_char, page_start, page_end, section, text, text_sha256, embedding, "
            "embedding_model, is_current) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true)",
            [
                (
                    tenant_id,
                    document_id,
                    version["id"],
                    document["collection_id"],
                    draft.ordinal,
                    draft.start,
                    draft.end,
                    draft.page_start,
                    draft.page_end,
                    draft.section,
                    draft.text,
                    draft.text_sha256,
                    embeddings[draft.text_sha256],
                    embedder.model_id,
                )
                for draft in drafts
            ],
        )
    if previous_id is not None:
        await conn.execute(
            "update passages set is_current = false where version_id = %s", (previous_id,)
        )
    await conn.execute(
        "update documents set current_version_id = %s where id = %s", (version["id"], document_id)
    )

    n_new = len(missing)
    n_reused = len(drafts) - n_new
    INGEST_PASSAGES.labels(outcome="new").inc(n_new)
    INGEST_PASSAGES.labels(outcome="reused").inc(n_reused)
    return VersionResult(
        "created_version",
        version["id"],
        version["version_number"],
        previous_id,
        len(drafts),
        n_reused,
        n_new,
    )


async def _reusable_embeddings(
    conn: AsyncConnection, text_hashes: list[bytes], model_id: str
) -> dict[bytes, np.ndarray]:
    """Embeddings already stored for these passage hashes (RLS keeps the lookup in the tenant)."""
    cur = await conn.execute(
        "select distinct on (text_sha256) text_sha256, embedding from passages "
        "where text_sha256 = any(%s) and embedding_model = %s",
        (text_hashes, model_id),
    )
    return {bytes(row["text_sha256"]): row["embedding"].to_numpy() for row in await cur.fetchall()}
