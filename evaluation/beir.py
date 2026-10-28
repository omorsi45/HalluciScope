"""BEIR datasets: load the files, and ingest the corpus into a benchmark collection.

The ingestion goes through ``parse_text`` and ``create_version``, the same code path an upload
takes, so the benchmark measures the real passage builder, embedder, and index.
"""

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from backend.db.pool import Database
from backend.db.repos import documents, tenants
from backend.ingest.parsers.text import parse_text
from backend.ingest.service import create_version
from backend.services import Services

# Documents per transaction: a failure rolls back at most this many, and no transaction
# holds a connection for a whole corpus.
BATCH_SIZE = 64


@dataclass
class BeirData:
    name: str
    corpus: dict[str, tuple[str, str]]  # BEIR id -> (title, text)
    queries: dict[str, str]  # query id -> text, only queries that have qrels
    qrels: dict[str, dict[str, int]]  # query id -> BEIR id -> relevance grade


@dataclass(frozen=True)
class IngestSummary:
    doc_map: dict[str, UUID]  # BEIR id -> document id, for existing and new documents
    n_ingested: int
    n_existing: int
    n_skipped_empty: int


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        # Text mode turns the CRLF line endings of the published files into "\n".
        if handle.readline().rstrip("\n").split("\t")[0] != "query-id":
            raise ValueError("qrels file does not start with the header row")
        for line in handle:
            line = line.rstrip("\n")
            if line:
                query_id, corpus_id, score = line.split("\t")
                qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def load_beir(root: Path, name: str, split: str = "test") -> BeirData:
    """Read a BEIR dataset folder: ``corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``.

    ``root`` is either the folder itself or its parent (a BEIR zip holds one top-level folder
    named after the dataset, so ``fetch(name)`` returns the parent).
    """
    folder = root if (root / "corpus.jsonl").is_file() else root / name
    qrels = _read_qrels(folder / "qrels" / f"{split}.tsv")
    corpus = {
        row["_id"]: (row.get("title") or "", row.get("text") or "")
        for row in _read_jsonl(folder / "corpus.jsonl")
    }
    queries = {
        row["_id"]: row["text"]
        for row in _read_jsonl(folder / "queries.jsonl")
        if row["_id"] in qrels
    }
    return BeirData(name, corpus, queries, qrels)


async def ensure_bench_scope(
    db: Database, tenant_name: str, collection_name: str
) -> tuple[UUID, UUID]:
    """The ids of a tenant and a ``benchmark`` collection in it, created when missing."""
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select id from tenants where name = %s order by created_at, id limit 1",
            (tenant_name,),
        )
        row = await cur.fetchone()
        if row is None:
            row = await tenants.insert_tenant(conn, tenant_name)
        tenant_id: UUID = row["id"]

    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select id, kind from collections where name = %s", (collection_name,)
        )
        row = await cur.fetchone()
        if row is None:
            cur = await conn.execute(
                "insert into collections (tenant_id, name, kind) "
                "values (%s, %s, 'benchmark') returning id",
                (tenant_id, collection_name),
            )
            row = await cur.fetchone()
        elif row["kind"] != "benchmark":
            raise ValueError(f"collection {collection_name!r} exists and is not a benchmark")
    return tenant_id, row["id"]


async def _existing_documents(
    conn: AsyncConnection, collection_id: UUID, filenames: list[str]
) -> dict[str, UUID]:
    cur = await conn.execute(
        "select filename, id from documents "
        "where collection_id = %s and filename = any(%s) and deleted_at is null "
        "order by created_at, id",
        (collection_id, filenames),
    )
    existing: dict[str, UUID] = {}
    for row in await cur.fetchall():
        existing.setdefault(row["filename"], row["id"])
    return existing


async def ingest_beir(
    services: Services,
    data: BeirData,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    doc_ids: Iterable[str] | None = None,
    filename_prefix: str = "",
) -> IngestSummary:
    """Ingest corpus documents (all, or ``doc_ids``) as one version each.

    The filename ``<prefix><beir_id>.txt`` identifies a document already in the collection, so a
    rerun adds only what is missing. Documents without text are skipped and counted.
    """
    ids = list(dict.fromkeys(data.corpus if doc_ids is None else doc_ids))
    effective_from = datetime.now(UTC)
    doc_map: dict[str, UUID] = {}
    n_ingested = n_existing = n_skipped_empty = 0

    for first in range(0, len(ids), BATCH_SIZE):
        batch = ids[first : first + BATCH_SIZE]
        filenames = {beir_id: f"{filename_prefix}{beir_id}.txt" for beir_id in batch}
        async with services.db.tenant_tx(tenant_id) as conn:
            existing = await _existing_documents(conn, collection_id, list(filenames.values()))
            for beir_id in batch:
                if filenames[beir_id] in existing:
                    doc_map[beir_id] = existing[filenames[beir_id]]
                    n_existing += 1
                    continue
                title, text = data.corpus[beir_id]
                raw = f"{title}\n\n{text}".encode() if title else text.encode()
                parsed = parse_text(raw)
                if not parsed.text:
                    n_skipped_empty += 1
                    continue
                document_id = await documents.insert_document(
                    conn,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    title=title or beir_id,
                    filename=filenames[beir_id],
                    mime_type="text/plain",
                    created_by=None,
                )
                # A benchmark document has no uploaded original, so no blob row is stored.
                await create_version(
                    conn,
                    services.embedder,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    parsed=parsed,
                    blob_sha256=hashlib.sha256(raw).digest(),
                    effective_from=effective_from,
                    created_by=None,
                    settings=services.settings,
                )
                doc_map[beir_id] = document_id
                n_ingested += 1
    return IngestSummary(doc_map, n_ingested, n_existing, n_skipped_empty)
