"""BM25 in SQL: reference parity, statistics over the visible set, deletion, and as-of rules."""

import asyncio
import hashlib
import itertools
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime

import pytest

from backend.db.repos import documents, tenants
from backend.ingest.parsers.text import parse_text
from backend.ingest.service import VersionResult, create_version
from backend.retrieval.bm25 import bm25_reference, bm25_search, bm25_term_score, query_terms
from backend.retrieval.types import SearchScope
from backend.retrieval.visible import visible_version_ids
from tests.pg_fixtures import seed_docs

pg = pytest.mark.pg

D_JAN = datetime(2026, 1, 1, tzinfo=UTC)
D_JUN = datetime(2026, 6, 1, tzinfo=UTC)
D_JUL = datetime(2026, 7, 1, tzinfo=UTC)

# Every word is at least six letters and not an English stopword, so a 60-word paragraph is
# over 400 characters: two paragraphs never fit one 800-character passage and each stays alone.
VOCAB = [
    "harbor", "copper", "lantern", "orchard", "granite", "meadow", "falcon", "velvet",
    "ember", "quartz", "willow", "saddle", "timber", "marble", "pepper", "anchor",
    "bridge", "candle", "dagger", "engine", "forest", "garnet", "hammer", "island",
    "jungle", "kettle", "ladder", "mirror", "needle", "oyster",
]  # fmt: skip


def make_corpus(seed: int = 7, docs: int = 3, paragraphs: int = 4) -> list[str]:
    """Documents of `paragraphs` paragraphs with varied length and vocabulary, so that tf, df,
    and document length all differ between passages."""
    rng = random.Random(seed)
    texts = []
    for _ in range(docs):
        parts = []
        for _ in range(paragraphs):
            pool = rng.sample(VOCAB, 12)
            words = [rng.choice(pool) for _ in range(rng.randint(60, 100))]
            parts.append(" ".join(words).capitalize() + ".")
        texts.append("\n\n".join(parts))
    return texts


QUERIES = [
    "harbor copper",
    "lantern",
    "the orchard of granite meadows",
    "velvet ember velvet ember quartz",
    "willow saddle timber marble pepper anchor bridge",
]


async def make_scope(db, *, collections: int = 1) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """A fresh tenant with `collections` empty collections."""
    async with db.system_tx() as conn:
        tenant_id = (await tenants.insert_tenant(conn, f"tenant-{uuid.uuid4().hex[:8]}"))["id"]
    ids = []
    async with db.tenant_tx(tenant_id) as conn:
        for _ in range(collections):
            cur = await conn.execute(
                "insert into collections (tenant_id, name) values (%s, %s) returning id",
                (tenant_id, f"docs-{uuid.uuid4().hex[:8]}"),
            )
            ids.append((await cur.fetchone())["id"])
    return tenant_id, ids


async def add_version(
    db, services, tenant_id, document_id, text: str, effective_from: datetime
) -> VersionResult:
    data = text.encode()
    async with db.tenant_tx(tenant_id) as conn:
        return await create_version(
            conn,
            services.embedder,
            tenant_id=tenant_id,
            document_id=document_id,
            parsed=parse_text(data),
            blob_sha256=hashlib.sha256(data).digest(),
            effective_from=effective_from,
            created_by=None,
            settings=services.settings,
        )


async def known_after(db, tenant_id) -> datetime:
    """The latest creation instant so far, read from Postgres and not from Python's clock.

    Waits until the database clock is strictly past it, so rows created afterwards get a larger
    created_at even if the host clock stepped back in between.
    """
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select greatest("
            "(select max(created_at) from document_versions), "
            "(select max(created_at) from documents), "
            "(select max(deleted_at) from documents)) as known"
        )
        known = (await cur.fetchone())["known"]
    for _ in range(400):
        async with db.tenant_tx(tenant_id) as conn:
            cur = await conn.execute("select clock_timestamp() > %s as ok", (known,))
            if (await cur.fetchone())["ok"]:
                return known
        await asyncio.sleep(0.05)
    pytest.fail("database clock did not pass the latest created_at", pytrace=False)


async def run_search(db, tenant_id, scope: SearchScope, query: str, *, limit: int = 100):
    async with db.tenant_tx(tenant_id) as conn:
        version_ids = await visible_version_ids(conn, scope)
        return await bm25_search(conn, query, scope, limit=limit, version_ids=version_ids)


async def version_of(db, tenant_id, passage_ids: list[uuid.UUID]) -> set[uuid.UUID]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select distinct version_id from passages where id = any(%s)", (passage_ids,)
        )
        return {row["version_id"] for row in await cur.fetchall()}


def test_term_score_literals():
    assert bm25_term_score(2, 1, 3, 10, 8) == pytest.approx(1.246538849981458, abs=1e-12)
    assert bm25_term_score(1, 3, 3, 5, 5) == pytest.approx(0.13353139262452254, abs=1e-12)


def test_reference_counts_every_passage_and_distinct_terms():
    # Three passages of length 4, 4, 8 (avgdl 16/3); only "a" is queried and it is in two.
    term_freqs = {"p1": {"a": 2}, "p2": {"a": 1, "b": 3}, "p3": {"b": 1}}
    doc_lens = {"p1": 4, "p2": 4, "p3": 8}

    scores = bm25_reference(["a", "a"], term_freqs, doc_lens)

    assert set(scores) == {"p1", "p2"}
    assert scores["p1"] == pytest.approx(bm25_term_score(2, 2, 3, 4, 16 / 3), abs=1e-15)
    assert scores["p2"] == pytest.approx(bm25_term_score(1, 2, 3, 4, 16 / 3), abs=1e-15)


@pg
async def test_sql_matches_reference(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, make_corpus())
    scope = SearchScope(tenant_id, (collection_id,))

    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select id, token_count from passages where is_current")
        doc_lens = {str(row["id"]): row["token_count"] for row in await cur.fetchall()}
        assert len(doc_lens) == 12

        for query in QUERIES:
            terms = await query_terms(conn, query)
            cur = await conn.execute(
                "select passage_id, term, tf from passage_terms where term = any(%s)", (terms,)
            )
            term_freqs: dict[str, dict[str, int]] = defaultdict(dict)
            for row in await cur.fetchall():
                term_freqs[str(row["passage_id"])][row["term"]] = row["tf"]
            expected = sorted(
                bm25_reference(terms, term_freqs, doc_lens).items(),
                key=lambda item: (-item[1], item[0]),
            )

            got = await bm25_search(conn, query, scope, limit=100, version_ids=None)

            assert expected, query
            gaps = [a[1] - b[1] for a, b in itertools.pairwise(expected)]
            assert all(gap > 1e-9 for gap in gaps), f"corpus has a score tie for {query!r}"
            assert [str(passage_id) for passage_id, _ in got] == [pid for pid, _ in expected]
            for (_, sql_score), (_, ref_score) in zip(got, expected, strict=True):
                assert sql_score == pytest.approx(ref_score, abs=1e-9, rel=0)

            top = await bm25_search(conn, query, scope, limit=3, version_ids=None)
            assert top == got[:3]


@pg
async def test_query_terms_are_distinct_sorted_lexemes(db):
    tenant_id, _ = await make_scope(db)
    async with db.tenant_tx(tenant_id) as conn:
        assert await query_terms(conn, "Velvet velvet EMBER the granite") == [
            "ember",
            "granit",
            "velvet",
        ]
        assert await query_terms(conn, "the and of") == []


@pg
async def test_stats_only_over_visible_set(db, services):
    tenant_id, (x, y) = await make_scope(db, collections=2)
    await seed_docs(db, services, tenant_id, x, make_corpus())
    scope_x = SearchScope(tenant_id, (x,))
    query = "harbor copper orchard"
    before = await run_search(db, tenant_id, scope_x, query)
    assert before

    outside = [f"Harbor copper orchard shipment {n} arrived at the dock." for n in range(50)]
    await seed_docs(db, services, tenant_id, y, outside)
    after = await run_search(db, tenant_id, scope_x, query)

    assert [passage_id for passage_id, _ in after] == [passage_id for passage_id, _ in before]
    for (_, score_after), (_, score_before) in zip(after, before, strict=True):
        assert score_after == pytest.approx(score_before, abs=1e-12, rel=0)

    # Control: the same query with Y in scope does change X's scores, so the check above
    # can detect leaked statistics.
    both = dict(await run_search(db, tenant_id, SearchScope(tenant_id, (x, y)), query))
    assert any(abs(both[passage_id] - score) > 1e-3 for passage_id, score in before)


@pg
async def test_deleted_documents_excluded(db, services):
    tenant_id, (x, z) = await make_scope(db, collections=2)
    texts = make_corpus()
    ids = await seed_docs(db, services, tenant_id, x, texts)
    # Z holds only the documents that stay live in X, so its statistics are the target.
    await seed_docs(db, services, tenant_id, z, [texts[0], texts[2]])
    query = "harbor copper orchard"
    async with db.tenant_tx(tenant_id) as conn:
        assert await documents.soft_delete_document(conn, ids[1]) is True
        cur = await conn.execute("select id from passages where document_id = %s", (ids[1],))
        deleted_passages = {row["id"] for row in await cur.fetchall()}
    assert len(deleted_passages) == 4

    live = await run_search(db, tenant_id, SearchScope(tenant_id, (x,)), query)
    reference = await run_search(db, tenant_id, SearchScope(tenant_id, (z,)), query)

    assert live
    assert not deleted_passages & {passage_id for passage_id, _ in live}
    assert [score for _, score in live] == pytest.approx(
        [score for _, score in reference], abs=1e-12, rel=0
    )
    # A scope that knows about the deletion excludes the document too.
    known = await known_after(db, tenant_id)
    temporal = await run_search(db, tenant_id, SearchScope(tenant_id, (x,), known_at=known), query)
    assert not deleted_passages & {passage_id for passage_id, _ in temporal}
    assert len(temporal) == len(live)


@pg
async def test_stopword_only_query_empty(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, make_corpus())

    hits = await run_search(db, tenant_id, SearchScope(tenant_id, (collection_id,)), "the and of")

    assert hits == []


@pg
async def test_empty_scope_returns_nothing(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, make_corpus())

    assert await run_search(db, tenant_id, SearchScope(tenant_id, ()), "harbor") == []
    temporal = SearchScope(tenant_id, (), as_of=D_JUN)
    assert await run_search(db, tenant_id, temporal, "harbor") == []


@pg
async def test_as_of_versions(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    (document_id,) = await seed_docs(
        db, services, tenant_id, collection_id, ["Harbor copper lantern."], effective_from=D_JAN
    )
    v2 = await add_version(db, services, tenant_id, document_id, "Harbor copper orchard.", D_JUN)
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select id from document_versions where version_number = 1 and document_id = %s",
            (document_id,),
        )
        v1_id = (await cur.fetchone())["id"]

    def scope(**kwargs) -> SearchScope:
        return SearchScope(tenant_id, (collection_id,), **kwargs)

    async def visible(search_scope: SearchScope):
        async with db.tenant_tx(tenant_id) as conn:
            return await visible_version_ids(conn, search_scope)

    async def search_versions(search_scope: SearchScope) -> set[uuid.UUID]:
        hits = await run_search(db, tenant_id, search_scope, "harbor copper")
        return await version_of(db, tenant_id, [passage_id for passage_id, _ in hits])

    # Before the first version took effect the document is invisible.
    before_first = scope(as_of=datetime(2025, 12, 1, tzinfo=UTC))
    assert await visible(before_first) == []
    assert await search_versions(before_first) == set()
    # Between the versions the older one answers, although it is no longer current.
    between = scope(as_of=datetime(2026, 3, 1, tzinfo=UTC))
    assert await visible(between) == [v1_id]
    assert await search_versions(between) == {v1_id}
    # Live scopes use the current version and need no version list.
    assert await visible(scope()) is None
    assert scope().is_live
    assert await search_versions(scope()) == {v2.version_id}

    known = await known_after(db, tenant_id)
    v3 = await add_version(db, services, tenant_id, document_id, "Harbor copper granite.", D_JUL)
    august = datetime(2026, 8, 1, tzinfo=UTC)
    # A version created after `known_at` does not exist for that scope, whatever its
    # effective_from says; without known_at it is the answer.
    assert await visible(scope(as_of=august, known_at=known)) == [v2.version_id]
    assert await search_versions(scope(as_of=august, known_at=known)) == {v2.version_id}
    assert await visible(scope(as_of=august)) == [v3.version_id]
    assert await search_versions(scope(as_of=august)) == {v3.version_id}
    assert not scope(as_of=august).is_live
    assert not scope(known_at=known).is_live


@pg
async def test_deleted_after_known_at_still_visible(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    (document_id,) = await seed_docs(
        db, services, tenant_id, collection_id, ["Harbor copper lantern."], effective_from=D_JAN
    )
    known = await known_after(db, tenant_id)
    async with db.tenant_tx(tenant_id) as conn:
        assert await documents.soft_delete_document(conn, document_id) is True

    live = await run_search(db, tenant_id, SearchScope(tenant_id, (collection_id,)), "harbor")
    as_known = await run_search(
        db, tenant_id, SearchScope(tenant_id, (collection_id,), known_at=known), "harbor"
    )
    after_delete = await known_after(db, tenant_id)
    as_later = await run_search(
        db,
        tenant_id,
        SearchScope(tenant_id, (collection_id,), known_at=after_delete),
        "harbor",
    )

    assert live == []
    assert len(as_known) == 1
    assert as_later == []
