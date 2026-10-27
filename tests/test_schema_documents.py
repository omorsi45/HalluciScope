"""Schema tests for documents, versions, passages, and their integrity triggers."""

import hashlib
import uuid
from dataclasses import dataclass

import numpy as np
import psycopg
import pytest

from tests.test_rls import GLOBAL_TABLES, TENANT_POLICY

pytestmark = pytest.mark.pg

NEW_TABLES = {
    "blobs",
    "documents",
    "document_versions",
    "passages",
    "passage_terms",
    "uploads",
    "search_requests",
}


@dataclass(frozen=True)
class Seeded:
    tenant_id: uuid.UUID
    collection_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    text: str


def vector(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(384, dtype=np.float32)


def sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


async def create_tenant(db, name: str) -> uuid.UUID:
    async with db.system_tx() as conn:
        cur = await conn.execute("insert into tenants (name) values (%s) returning id", (name,))
        return (await cur.fetchone())["id"]


async def insert_collection(conn, tenant_id: uuid.UUID) -> uuid.UUID:
    cur = await conn.execute(
        "insert into collections (tenant_id, name) values (%s, %s) returning id",
        (tenant_id, f"docs-{uuid.uuid4().hex[:8]}"),
    )
    return (await cur.fetchone())["id"]


async def insert_document(conn, tenant_id: uuid.UUID, collection_id: uuid.UUID) -> uuid.UUID:
    cur = await conn.execute(
        "insert into documents (tenant_id, collection_id, title, filename, mime_type) "
        "values (%s, %s, 'Policy', 'policy.txt', 'text/plain') returning id",
        (tenant_id, collection_id),
    )
    return (await cur.fetchone())["id"]


async def insert_version(db, text: str, *, tenant_id: uuid.UUID | None = None) -> Seeded:
    """Create a tenant (unless given), collection, document, and version 1 directly in SQL."""
    if tenant_id is None:
        tenant_id = await create_tenant(db, "acme")
    async with db.tenant_tx(tenant_id) as conn:
        collection_id = await insert_collection(conn, tenant_id)
        document_id = await insert_document(conn, tenant_id, collection_id)
        cur = await conn.execute(
            "insert into document_versions (tenant_id, document_id, version_number, "
            "effective_from, content_sha256, blob_sha256, text, parser, parser_version, "
            "char_count) values (%s, %s, 1, now(), %s, %s, %s, 'text', '1', %s) returning id",
            (tenant_id, document_id, sha256(text), sha256(text), text, len(text)),
        )
        version_id = (await cur.fetchone())["id"]
        await conn.execute(
            "update documents set current_version_id = %s where id = %s", (version_id, document_id)
        )
    return Seeded(tenant_id, collection_id, document_id, version_id, text)


async def insert_passage(
    conn,
    seeded: Seeded,
    text: str,
    start: int,
    end: int,
    *,
    ordinal: int = 0,
    document_id: uuid.UUID | None = None,
    text_sha256: bytes | None = None,
    embedding: np.ndarray | None = None,
) -> uuid.UUID:
    cur = await conn.execute(
        "insert into passages (tenant_id, document_id, version_id, collection_id, ordinal, "
        "start_char, end_char, text, text_sha256, embedding, embedding_model) "
        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'test-model') returning id",
        (
            seeded.tenant_id,
            document_id or seeded.document_id,
            seeded.version_id,
            seeded.collection_id,
            ordinal,
            start,
            end,
            text,
            text_sha256 or sha256(text),
            vector() if embedding is None else embedding,
        ),
    )
    return (await cur.fetchone())["id"]


async def test_passage_must_match_slice(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    async with db.tenant_tx(seeded.tenant_id) as conn:
        await insert_passage(conn, seeded, "Refunds", 0, 7)
    with pytest.raises(psycopg.errors.RaiseException, match="does not match version slice"):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await insert_passage(conn, seeded, "Refundz", 0, 7, ordinal=1)


async def test_passage_hash_must_match_text(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    with pytest.raises(psycopg.errors.RaiseException, match="does not match version slice"):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await insert_passage(conn, seeded, "Refunds", 0, 7, text_sha256=sha256("Refundz"))


async def test_unicode_slice_matches(db):
    text = "Café 😀 naïve 東京 ok"
    seeded = await insert_version(db, text)
    async with db.tenant_tx(seeded.tenant_id) as conn:
        passage_id = await insert_passage(conn, seeded, text[5:12], 5, 12)
        cur = await conn.execute("select text from passages where id = %s", (passage_id,))
        assert (await cur.fetchone())["text"] == "😀 naïve"


async def test_terms_and_token_count_filled(db):
    text = "Refunds are refunded within 30 days"
    seeded = await insert_version(db, text)
    async with db.tenant_tx(seeded.tenant_id) as conn:
        passage_id = await insert_passage(conn, seeded, text, 0, len(text))
        cur = await conn.execute("select token_count from passages where id = %s", (passage_id,))
        token_count = (await cur.fetchone())["token_count"]
        cur = await conn.execute(
            "select term, tf from passage_terms where passage_id = %s", (passage_id,)
        )
        terms = {row["term"]: row["tf"] for row in await cur.fetchall()}
    assert token_count == 5
    assert terms["refund"] == 2
    assert sum(terms.values()) == token_count


async def test_versions_immutable(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    with pytest.raises(psycopg.errors.RaiseException, match="document_versions are immutable"):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await conn.execute(
                "update document_versions set text = 'x' where id = %s", (seeded.version_id,)
            )
    with pytest.raises(psycopg.errors.RaiseException, match="document_versions are immutable"):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await conn.execute("delete from document_versions where id = %s", (seeded.version_id,))


async def test_passage_content_immutable_but_flags_mutable(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    async with db.tenant_tx(seeded.tenant_id) as conn:
        passage_id = await insert_passage(conn, seeded, "Refunds", 0, 7)
    with pytest.raises(psycopg.errors.RaiseException, match="passage content is immutable"):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await conn.execute("update passages set text = 'Refundz' where id = %s", (passage_id,))
    async with db.tenant_tx(seeded.tenant_id) as conn:
        other_collection = await insert_collection(conn, seeded.tenant_id)
        await conn.execute(
            "update passages set is_current = true, collection_id = %s where id = %s",
            (other_collection, passage_id),
        )
        cur = await conn.execute(
            "select is_current, collection_id from passages where id = %s", (passage_id,)
        )
        row = await cur.fetchone()
    assert (row["is_current"], row["collection_id"]) == (True, other_collection)


async def test_passage_delete_rejected(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    async with db.tenant_tx(seeded.tenant_id) as conn:
        passage_id = await insert_passage(conn, seeded, "Refunds", 0, 7)
    with pytest.raises(psycopg.errors.RaiseException, match="passage content is immutable"):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await conn.execute("delete from passages where id = %s", (passage_id,))


async def test_passage_version_document_mismatch_rejected(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    async with db.tenant_tx(seeded.tenant_id) as conn:
        other_document = await insert_document(conn, seeded.tenant_id, seeded.collection_id)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        async with db.tenant_tx(seeded.tenant_id) as conn:
            await insert_passage(conn, seeded, "Refunds", 0, 7, document_id=other_document)


@pytest.mark.parametrize("child", ["document", "passage"])
async def test_row_pointing_at_other_tenants_collection_rejected(db, child):
    seeded_a = await insert_version(db, "Refunds take 30 days.")
    tenant_b = await create_tenant(db, "globex")
    seeded_b = await insert_version(db, "Refunds take 30 days.", tenant_id=tenant_b)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        async with db.tenant_tx(tenant_b) as conn:
            if child == "document":
                await insert_document(conn, tenant_b, seeded_a.collection_id)
            else:
                foreign = Seeded(
                    tenant_b,
                    seeded_a.collection_id,
                    seeded_b.document_id,
                    seeded_b.version_id,
                    seeded_b.text,
                )
                await insert_passage(conn, foreign, "Refunds", 0, 7)


async def test_new_tables_have_forced_rls(db):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select c.relname, c.relrowsecurity, c.relforcerowsecurity "
            "from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where c.relkind in ('r', 'p') and n.nspname = 'public'"
        )
        tables = {row["relname"]: row for row in await cur.fetchall()}
        cur = await conn.execute(
            "select tablename, qual, with_check from pg_policies "
            "where schemaname = 'public' and policyname = 'tenant_isolation'"
        )
        policies = {row["tablename"]: row for row in await cur.fetchall()}

    assert tables.keys() >= NEW_TABLES
    assert NEW_TABLES.isdisjoint(GLOBAL_TABLES)
    unprotected = sorted(
        name
        for name in NEW_TABLES
        if not (tables[name]["relrowsecurity"] and tables[name]["relforcerowsecurity"])
    )
    assert unprotected == []
    assert sorted(NEW_TABLES - policies.keys()) == []
    wrong_policy = sorted(
        name
        for name in NEW_TABLES
        if (policies[name]["qual"], policies[name]["with_check"]) != (TENANT_POLICY, TENANT_POLICY)
    )
    assert wrong_policy == []


async def test_embedding_roundtrip(db):
    seeded = await insert_version(db, "Refunds take 30 days.")
    expected = vector(seed=7)
    async with db.tenant_tx(seeded.tenant_id) as conn:
        passage_id = await insert_passage(conn, seeded, "Refunds", 0, 7, embedding=expected)
        cur = await conn.execute("select embedding from passages where id = %s", (passage_id,))
        # pgvector 0.5.1 loads a vector column as pgvector.Vector, not an ndarray.
        actual = (await cur.fetchone())["embedding"].to_numpy()
    assert actual.dtype == np.float32
    assert actual.shape == (384,)
    assert np.array_equal(actual, expected)


async def test_plan_a_deferred_indexes_exist(db):
    expected = [
        ("memberships", "(user_id)"),
        ("collection_grants", "(tenant_id, user_id)"),
        ("sessions", "(tenant_id, user_id)"),
        ("api_keys", "(tenant_id, user_id)"),
    ]
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select tablename, indexdef from pg_indexes where schemaname = 'public'"
        )
        definitions = [(row["tablename"], row["indexdef"]) for row in await cur.fetchall()]
    missing = [
        (table, columns)
        for table, columns in expected
        if not any(t == table and d.endswith(columns) for t, d in definitions)
    ]
    assert missing == []
