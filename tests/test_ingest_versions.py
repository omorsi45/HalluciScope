"""Version creation: dedupe, incremental embeddings, offsets, limits, and the documents repo."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from backend.api.pagination import PageParams
from backend.config import Settings
from backend.db.repos import documents, tenants
from backend.ingest import service
from backend.ingest.errors import IngestError
from backend.ingest.service import VersionResult, create_version
from backend.ingest.types import PageSpan, ParsedDocument, SectionSpan
from backend.obs.metrics import INGEST_PASSAGES
from tests.fakes import HashEmbedder

pytestmark = pytest.mark.pg

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SETTINGS = Settings()


class CountingEmbedder:
    """HashEmbedder that records every text it is asked to embed."""

    def __init__(self) -> None:
        self._inner = HashEmbedder()
        self.model_id = self._inner.model_id
        self.dim = self._inner.dim
        self.calls: list[list[str]] = []

    def embed_passages(self, texts):
        self.calls.append(list(texts))
        return self._inner.embed_passages(texts)

    def embed_queries(self, texts):
        return self._inner.embed_queries(texts)

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


@dataclass(frozen=True)
class Seeded:
    tenant_id: uuid.UUID
    collection_id: uuid.UUID
    document_id: uuid.UUID


PARAGRAPHS = [
    "Refunds are issued within 30 days. The customer must keep the receipt. "
    "Digital goods are excluded.",
    "Shipping takes five business days. Express shipping takes two days. "
    "Orders to islands take longer.",
    "Support is open on weekdays. Weekend requests are answered on Monday. "
    "Urgent issues can call the hotline.",
    "Warranties last one year. Batteries are covered for six months. "
    "Accidental damage is not covered.",
    "Accounts can be closed at any time. Closing removes saved addresses. "
    "Invoices stay available for seven years.",
]


def parse_paragraphs(paragraphs: list[str]) -> ParsedDocument:
    """One section per paragraph, so every paragraph becomes exactly one passage."""
    text = "\n\n".join(paragraphs)
    sections = []
    cursor = 0
    for index, paragraph in enumerate(paragraphs):
        sections.append(SectionSpan(cursor, cursor + len(paragraph), f"Part {index + 1}"))
        cursor += len(paragraph) + 2
    return ParsedDocument(text, (), tuple(sections), {}, "text", "1")


def parse_plain(text: str) -> ParsedDocument:
    return ParsedDocument(text, (), (), {}, "text", "1")


def blob_hash(parsed: ParsedDocument) -> bytes:
    return hashlib.sha256(parsed.text.encode()).digest()


async def seed(db, name: str = "acme") -> Seeded:
    async with db.system_tx() as conn:
        tenant_id = (await tenants.insert_tenant(conn, name))["id"]
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into collections (tenant_id, name) values (%s, %s) returning id",
            (tenant_id, f"docs-{uuid.uuid4().hex[:8]}"),
        )
        collection_id = (await cur.fetchone())["id"]
        document_id = await documents.insert_document(
            conn,
            tenant_id=tenant_id,
            collection_id=collection_id,
            title="Policy",
            filename="policy.txt",
            mime_type="text/plain",
            created_by=None,
        )
    return Seeded(tenant_id, collection_id, document_id)


async def ingest(
    db,
    embedder,
    seeded: Seeded,
    parsed: ParsedDocument,
    *,
    effective_from: datetime = T0,
) -> VersionResult:
    async with db.tenant_tx(seeded.tenant_id) as conn:
        return await create_version(
            conn,
            embedder,
            tenant_id=seeded.tenant_id,
            document_id=seeded.document_id,
            parsed=parsed,
            blob_sha256=blob_hash(parsed),
            effective_from=effective_from,
            created_by=None,
            settings=SETTINGS,
        )


async def fetch_all(db, seeded: Seeded, query: str, params: tuple = ()) -> list[dict]:
    async with db.tenant_tx(seeded.tenant_id) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def test_first_version_creates_passages(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    parsed = parse_paragraphs(PARAGRAPHS)
    new_before = INGEST_PASSAGES.labels(outcome="new").value
    reused_before = INGEST_PASSAGES.labels(outcome="reused").value

    result = await ingest(db, embedder, seeded, parsed)

    assert result.status == "created_version"
    assert result.version_number == 1
    assert result.previous_version_id is None
    assert result.n_passages == 5
    assert result.n_new == result.n_passages
    assert result.n_reused == 0
    assert INGEST_PASSAGES.labels(outcome="new").value - new_before == 5
    assert INGEST_PASSAGES.labels(outcome="reused").value - reused_before == 0
    rows = await fetch_all(
        db,
        seeded,
        "select ordinal, is_current, embedding_model from passages where version_id = %s "
        "order by ordinal",
        (result.version_id,),
    )
    assert [row["ordinal"] for row in rows] == [0, 1, 2, 3, 4]
    assert all(row["is_current"] for row in rows)
    assert {row["embedding_model"] for row in rows} == {embedder.model_id}
    (document,) = await fetch_all(
        db, seeded, "select current_version_id from documents where id = %s", (seeded.document_id,)
    )
    assert document["current_version_id"] == result.version_id
    (version,) = await fetch_all(
        db,
        seeded,
        "select char_count, content_sha256, sections from document_versions where id = %s",
        (result.version_id,),
    )
    assert version["char_count"] == len(parsed.text)
    assert version["content_sha256"] == hashlib.sha256(parsed.text.encode()).digest()
    assert version["sections"][0] == [0, len(PARAGRAPHS[0]), "Part 1"]


async def test_page_map_is_stored_as_triples(db):
    seeded = await seed(db)
    text = "First page text here. Second page text there."
    parsed = ParsedDocument(
        text,
        (PageSpan(1, 0, 21), PageSpan(2, 22, len(text))),
        (),
        {"title": "Doc"},
        "pdf",
        "9",
    )

    result = await ingest(db, CountingEmbedder(), seeded, parsed)

    (version,) = await fetch_all(
        db,
        seeded,
        "select page_map, metadata, parser, parser_version from document_versions where id = %s",
        (result.version_id,),
    )
    assert version["page_map"] == [[1, 0, 21], [2, 22, len(text)]]
    assert version["metadata"] == {"title": "Doc"}
    assert (version["parser"], version["parser_version"]) == ("pdf", "9")


async def test_identical_text_is_duplicate(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    parsed = parse_paragraphs(PARAGRAPHS)
    first = await ingest(db, embedder, seeded, parsed)
    embedded_before = len(embedder.embedded_texts)

    second = await ingest(db, embedder, seeded, parsed)

    assert second.status == "duplicate"
    assert second.version_id == first.version_id
    assert second.version_number == 1
    assert len(embedder.embedded_texts) == embedded_before
    (versions,) = await fetch_all(db, seeded, "select count(*) as n from document_versions")
    (passages,) = await fetch_all(db, seeded, "select count(*) as n from passages")
    assert (versions["n"], passages["n"]) == (1, 5)


async def test_changed_version_reuses_unchanged_passages(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    first = await ingest(db, embedder, seeded, parse_paragraphs(PARAGRAPHS))
    embedded_before = len(embedder.embedded_texts)
    changed = list(PARAGRAPHS)
    changed[2] = "Support is open every day. Urgent issues can call the hotline."
    new_before = INGEST_PASSAGES.labels(outcome="new").value
    reused_before = INGEST_PASSAGES.labels(outcome="reused").value

    second = await ingest(
        db, embedder, seeded, parse_paragraphs(changed), effective_from=T0 + timedelta(days=1)
    )

    assert second.status == "created_version"
    assert second.version_number == 2
    assert second.previous_version_id == first.version_id
    assert second.n_reused >= 3
    assert second.n_new + second.n_reused == second.n_passages
    assert len(embedder.embedded_texts) - embedded_before == second.n_new
    assert embedder.embedded_texts[embedded_before:] == [changed[2]]
    assert INGEST_PASSAGES.labels(outcome="new").value - new_before == second.n_new
    assert INGEST_PASSAGES.labels(outcome="reused").value - reused_before == second.n_reused
    flags = await fetch_all(
        db,
        seeded,
        "select version_id, bool_and(is_current) as cur, bool_or(is_current) as anyc "
        "from passages group by version_id",
    )
    by_version = {row["version_id"]: row for row in flags}
    assert by_version[first.version_id]["anyc"] is False
    assert by_version[second.version_id]["cur"] is True
    (document,) = await fetch_all(
        db, seeded, "select current_version_id from documents where id = %s", (seeded.document_id,)
    )
    assert document["current_version_id"] == second.version_id


async def test_reused_embedding_equals_original(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    first = await ingest(db, embedder, seeded, parse_paragraphs(PARAGRAPHS))
    changed = list(PARAGRAPHS)
    changed[4] = "Accounts can be closed at any time."
    second = await ingest(
        db, embedder, seeded, parse_paragraphs(changed), effective_from=T0 + timedelta(days=1)
    )

    rows = await fetch_all(
        db,
        seeded,
        "select version_id, text, embedding from passages where text = %s",
        (PARAGRAPHS[0],),
    )

    assert {row["version_id"] for row in rows} == {first.version_id, second.version_id}
    first_vector, second_vector = (row["embedding"].to_numpy() for row in rows)
    assert (first_vector == second_vector).all()
    assert (first_vector == embedder._inner.embed_passages([PARAGRAPHS[0]])[0]).all()


async def test_revert_creates_new_version(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    original = parse_paragraphs(PARAGRAPHS)
    edited = list(PARAGRAPHS)
    edited[1] = "Shipping takes ten business days."
    await ingest(db, embedder, seeded, original)
    await ingest(db, embedder, seeded, parse_paragraphs(edited), effective_from=T0 + timedelta(1))
    embedded_before = len(embedder.embedded_texts)

    third = await ingest(db, embedder, seeded, original, effective_from=T0 + timedelta(days=2))

    assert third.status == "created_version"
    assert third.version_number == 3
    assert third.n_new == 0
    assert third.n_reused == third.n_passages == 5
    assert len(embedder.embedded_texts) == embedded_before


async def test_effective_from_out_of_order(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    first = await ingest(
        db, embedder, seeded, parse_paragraphs(PARAGRAPHS), effective_from=T0 + timedelta(days=5)
    )
    changed = list(PARAGRAPHS)
    changed[0] = "Refunds are issued within 14 days."

    with pytest.raises(IngestError) as raised:
        await ingest(db, embedder, seeded, parse_paragraphs(changed), effective_from=T0)

    assert raised.value.code == "effective_from_out_of_order"
    (versions,) = await fetch_all(db, seeded, "select count(*) as n from document_versions")
    assert versions["n"] == 1
    (document,) = await fetch_all(
        db, seeded, "select current_version_id from documents where id = %s", (seeded.document_id,)
    )
    assert document["current_version_id"] == first.version_id
    current = await fetch_all(db, seeded, "select version_id from passages where is_current")
    assert {row["version_id"] for row in current} == {first.version_id}


UNICODE_TEXTS = {
    "precomposed": "Café 😀 opens at 9. 東京 office closes at 5. Naïve test.",
    "combining": "Cafe\u0301 😀 opens at 9. 東京 office closes at 5. Nai\u0308ve test.",
}


@pytest.mark.parametrize("text", UNICODE_TEXTS.values(), ids=UNICODE_TEXTS.keys())
async def test_unicode_offsets_end_to_end(db, text):
    seeded = await seed(db)
    split = text.index(" 東京")
    parsed = ParsedDocument(
        text,
        (),
        (SectionSpan(0, split, "Morning"), SectionSpan(split + 1, len(text), "Evening")),
        {},
        "text",
        "1",
    )

    result = await ingest(db, CountingEmbedder(), seeded, parsed)

    assert result.n_passages == 2
    async with db.tenant_tx(seeded.tenant_id) as conn:
        boundaries = await documents.list_passages(conn, result.version_id)
        stored = await (
            await conn.execute(
                "select ordinal, text from passages where version_id = %s order by ordinal",
                (result.version_id,),
            )
        ).fetchall()
        for boundary, row in zip(boundaries, stored, strict=True):
            piece = await documents.get_version_text(
                conn, result.version_id, boundary["start_char"], boundary["end_char"]
            )
            assert piece == row["text"]
            assert piece == text[boundary["start_char"] : boundary["end_char"]]
    assert boundaries[1]["start_char"] == split + 1


async def test_too_many_passages(db, monkeypatch):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    monkeypatch.setattr(service, "MAX_PASSAGES", 2)

    with pytest.raises(IngestError) as raised:
        await ingest(db, embedder, seeded, parse_paragraphs(PARAGRAPHS))

    assert raised.value.code == "too_many_passages"
    assert embedder.calls == []
    (versions,) = await fetch_all(db, seeded, "select count(*) as n from document_versions")
    assert versions["n"] == 0


@pytest.mark.parametrize("text", ["", "   \n\t  \n"], ids=["empty", "whitespace"])
async def test_no_text(db, text):
    embedder = CountingEmbedder()
    seeded = await seed(db)

    with pytest.raises(IngestError) as raised:
        await ingest(db, embedder, seeded, parse_plain(text))

    assert raised.value.code == "no_text"
    assert embedder.calls == []


async def test_missing_or_deleted_document_raises_lookup_error(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    missing = Seeded(seeded.tenant_id, seeded.collection_id, uuid.uuid4())
    with pytest.raises(LookupError):
        await ingest(db, embedder, missing, parse_paragraphs(PARAGRAPHS))

    async with db.tenant_tx(seeded.tenant_id) as conn:
        assert await documents.soft_delete_document(conn, seeded.document_id) is True
    with pytest.raises(LookupError):
        await ingest(db, embedder, seeded, parse_paragraphs(PARAGRAPHS))
    assert embedder.calls == []


async def test_reuse_does_not_cross_tenants(db):
    embedder = CountingEmbedder()
    tenant_a = await seed(db, "tenant-a")
    tenant_b = await seed(db, "tenant-b")
    parsed = parse_paragraphs(PARAGRAPHS)
    first = await ingest(db, embedder, tenant_a, parsed)
    assert first.n_new == 5
    embedded_before = len(embedder.embedded_texts)

    second = await ingest(db, embedder, tenant_b, parsed)

    assert second.status == "created_version"
    assert second.n_reused == 0
    assert second.n_new == second.n_passages == 5
    assert len(embedder.embedded_texts) - embedded_before == 5


async def test_embeddings_are_batched(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    small_batches = Settings(embed_batch_size=2)
    parsed = parse_paragraphs(PARAGRAPHS)

    async with db.tenant_tx(seeded.tenant_id) as conn:
        await create_version(
            conn,
            embedder,
            tenant_id=seeded.tenant_id,
            document_id=seeded.document_id,
            parsed=parsed,
            blob_sha256=blob_hash(parsed),
            effective_from=T0,
            created_by=None,
            settings=small_batches,
        )

    assert [len(call) for call in embedder.calls] == [2, 2, 1]


async def test_documents_repo_round_trip(db):
    embedder = CountingEmbedder()
    seeded = await seed(db)
    async with db.tenant_tx(seeded.tenant_id) as conn:
        other_id = await documents.insert_document(
            conn,
            tenant_id=seeded.tenant_id,
            collection_id=seeded.collection_id,
            title="Second",
            filename="second.md",
            mime_type="text/markdown",
            created_by=None,
        )
    first = await ingest(db, embedder, seeded, parse_paragraphs(PARAGRAPHS))
    edited = list(PARAGRAPHS)
    edited[0] = "Refunds are issued within 14 days."
    second = await ingest(
        db, embedder, seeded, parse_paragraphs(edited), effective_from=T0 + timedelta(days=1)
    )
    sha = hashlib.sha256(b"raw bytes").digest()

    async with db.tenant_tx(seeded.tenant_id) as conn:
        await documents.insert_blob(conn, seeded.tenant_id, sha, b"raw bytes")
        await documents.insert_blob(conn, seeded.tenant_id, sha, b"raw bytes")
        assert await documents.get_blob(conn, sha) == b"raw bytes"
        assert await documents.get_blob(conn, hashlib.sha256(b"nope").digest()) is None

        everything = await documents.list_documents(
            conn, [seeded.collection_id], PageParams(limit=5, after=None)
        )
        assert [row["id"] for row in everything] == [other_id, seeded.document_id]
        by_id = {row["id"]: row for row in everything}
        assert by_id[seeded.document_id]["current_version_number"] == 2
        assert by_id[seeded.document_id]["version_count"] == 2
        assert by_id[other_id]["current_version_number"] is None
        assert by_id[other_id]["version_count"] == 0
        assert (
            await documents.list_documents(conn, [uuid.uuid4()], PageParams(limit=5, after=None))
            == []
        )

        head = await documents.list_documents(
            conn, [seeded.collection_id], PageParams(limit=1, after=None)
        )
        assert len(head) == 2  # limit + 1 rows tell the caller another page exists
        tail = await documents.list_documents(
            conn,
            [seeded.collection_id],
            PageParams(limit=1, after=(head[0]["created_at"], str(head[0]["id"]))),
        )
        assert [row["id"] for row in tail] == [seeded.document_id]

        versions = await documents.list_versions(conn, seeded.document_id)
        assert [v["version_number"] for v in versions] == [1, 2]
        assert all("text" not in v for v in versions)
        assert versions[1]["id"] == second.version_id
        version = await documents.get_version(conn, seeded.document_id, 1)
        assert version["id"] == first.version_id
        assert version["text"] == "\n\n".join(PARAGRAPHS)
        assert await documents.get_version(conn, seeded.document_id, 9) is None

        passages = await documents.list_passages(conn, first.version_id)
        assert [p["ordinal"] for p in passages] == [0, 1, 2, 3, 4]
        assert all("text" not in p for p in passages)
        assert {"start_char", "end_char", "page_start", "page_end", "section"} <= set(passages[0])

        assert await documents.count_live_documents(conn, seeded.collection_id) == 2
        assert await documents.soft_delete_document(conn, seeded.document_id) is True
        assert await documents.soft_delete_document(conn, seeded.document_id) is False
        assert await documents.get_document(conn, seeded.document_id) is None
        deleted = await documents.get_document(conn, seeded.document_id, include_deleted=True)
        assert deleted["deleted_at"] is not None
        assert (await documents.get_document(conn, other_id))["title"] == "Second"
        assert await documents.count_live_documents(conn, seeded.collection_id) == 1
        remaining = await documents.list_documents(
            conn, [seeded.collection_id], PageParams(limit=5, after=None)
        )
        assert [row["id"] for row in remaining] == [other_id]
