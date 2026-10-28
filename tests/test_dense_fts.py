"""Filtered pgvector search and the Postgres full-text baseline over the visible passage set."""

import random
import uuid
from datetime import UTC, datetime

import pytest

from backend.retrieval.dense import dense_search
from backend.retrieval.fts import fts_search
from backend.retrieval.types import SearchScope
from backend.retrieval.visible import visible_version_ids
from tests.fakes import HashEmbedder
from tests.pg_fixtures import seed_docs
from tests.test_bm25 import D_JAN, D_JUN, VOCAB, add_version, make_scope, version_of

pg = pytest.mark.pg

EMBEDDER = HashEmbedder()


def query_vector(text: str):
    return EMBEDDER.embed_queries([text])[0]


async def run_dense(
    db,
    tenant_id,
    scope: SearchScope,
    query: str,
    *,
    limit: int = 10,
    exact: bool = False,
    force_index: bool = False,
):
    """``force_index`` makes the planner order by the HNSW index instead of joining documents
    first and sorting (it would on these tiny tables), so the index is the access path under
    test."""
    async with db.tenant_tx(tenant_id) as conn:
        if force_index:
            await conn.execute("set local enable_seqscan = off")
            await conn.execute("set local enable_sort = off")
        version_ids = await visible_version_ids(conn, scope)
        return await dense_search(
            conn, query_vector(query), scope, limit=limit, version_ids=version_ids, exact=exact
        )


async def run_fts(db, tenant_id, scope: SearchScope, query: str, *, limit: int = 100):
    async with db.tenant_tx(tenant_id) as conn:
        version_ids = await visible_version_ids(conn, scope)
        return await fts_search(conn, query, scope, limit=limit, version_ids=version_ids)


async def passage_ids_of(db, tenant_id, collection_id) -> set[uuid.UUID]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select p.id from passages p join documents d on d.id = p.document_id "
            "where d.collection_id = %s",
            (collection_id,),
        )
        return {row["id"] for row in await cur.fetchall()}


@pg
async def test_dense_nearest_first(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "Shipping takes five working days to arrive.",
        "The refund window is thirty days from delivery.",
        "Our office is closed on public holidays.",
        "Passwords must be rotated every ninety days.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))

    hits = await run_dense(db, tenant_id, scope, "refund window days")

    assert len(hits) == 4
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select text from passages where id = %s", (hits[0][0],))
        assert (await cur.fetchone())["text"] == texts[1]
    similarities = [similarity for _, similarity in hits]
    assert similarities == sorted(similarities, reverse=True)
    # 1 - cosine distance of unit vectors: the refund passage shares two of three query words.
    assert 0.4 < similarities[0] <= 1.0


@pg
async def test_exact_and_hnsw_agree(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    rng = random.Random(11)
    # Repeated words and a per-passage amount of filler give every passage its own norm and
    # weights, so similarities are not tied at rank 5 (a tie would let two correct searches
    # return different sets).
    texts = [
        " ".join([*rng.choices(VOCAB, k=rng.randint(8, 20)), *["filler"] * n, f"item{n}"]) + "."
        for n in range(40)
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))

    for _ in range(10):
        query = " ".join(rng.choices(VOCAB, k=4))
        approximate = await run_dense(db, tenant_id, scope, query, limit=5, force_index=True)
        exact = await run_dense(db, tenant_id, scope, query, limit=6, exact=True)

        assert len(exact) == 6
        assert exact[4][1] - exact[5][1] > 1e-6, f"corpus ties at rank 5 for {query!r}"
        assert [passage_id for passage_id, _ in approximate] == [
            passage_id for passage_id, _ in exact[:5]
        ], query
        for (_, approx_similarity), (_, exact_similarity) in zip(
            approximate, exact[:5], strict=True
        ):
            assert approx_similarity == pytest.approx(exact_similarity, abs=1e-6)


@pg
async def test_dense_filter_does_not_starve(db, services):
    tenant_id, (x, y) = await make_scope(db, collections=2)
    rng = random.Random(5)
    # X shares two query words with Y, so the HNSW graph links them; a far outlier can end up
    # with no incoming edge, which no scan setting can fix. Y holds all three query words and
    # fewer filler words, so it is closer to the query than every X passage.
    await seed_docs(
        db,
        services,
        tenant_id,
        x,
        [
            f"Refund window {' '.join(rng.choices(VOCAB, k=rng.randint(3, 6)))} memo{n}."
            for n in range(3)
        ],
    )
    await seed_docs(
        db,
        services,
        tenant_id,
        y,
        [
            f"Refund window days {' '.join(rng.choices(VOCAB, k=rng.randint(2, 4)))} case{n}."
            for n in range(300)
        ],
    )
    readable = await passage_ids_of(db, tenant_id, x)
    assert len(readable) == 3
    query = "refund window days"
    everything = SearchScope(tenant_id, (x, y))
    # Preconditions: the 100 passages the first index batch holds are all unreadable ones, and
    # the index reaches every passage when nothing is filtered out. Only the filter can starve.
    nearest = await run_dense(db, tenant_id, everything, query, limit=100, exact=True)
    assert not readable & {passage_id for passage_id, _ in nearest}
    reached = await run_dense(db, tenant_id, everything, query, limit=303, force_index=True)
    assert len(reached) == 303

    hits = await run_dense(
        db, tenant_id, SearchScope(tenant_id, (x,)), query, limit=3, force_index=True
    )

    assert {passage_id for passage_id, _ in hits} == readable


@pg
async def test_dense_respects_as_of(db, services):
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

    between = SearchScope(tenant_id, (collection_id,), as_of=datetime(2026, 3, 1, tzinfo=UTC))
    live = SearchScope(tenant_id, (collection_id,))

    older = await run_dense(db, tenant_id, between, "harbor copper")
    current = await run_dense(db, tenant_id, live, "harbor copper")

    assert len(older) == 1
    assert await version_of(db, tenant_id, [passage_id for passage_id, _ in older]) == {v1_id}
    assert len(current) == 1
    assert await version_of(db, tenant_id, [passage_id for passage_id, _ in current]) == {
        v2.version_id
    }


@pg
async def test_dense_session_settings(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Refund window days."])
    scope = SearchScope(tenant_id, (collection_id,))

    async def settings_after(limit: int, *, exact: bool = False) -> dict[str, str]:
        async with db.tenant_tx(tenant_id) as conn:
            await dense_search(
                conn, query_vector("refund"), scope, limit=limit, version_ids=None, exact=exact
            )
            cur = await conn.execute(
                "select current_setting('hnsw.ef_search') as ef_search, "
                "current_setting('hnsw.iterative_scan') as iterative_scan, "
                "current_setting('enable_indexscan') as indexscan, "
                "current_setting('enable_bitmapscan') as bitmapscan"
            )
            return dict(await cur.fetchone())

    assert await settings_after(10) == {
        "ef_search": "100",
        "iterative_scan": "relaxed_order",
        "indexscan": "on",
        "bitmapscan": "on",
    }
    assert (await settings_after(250))["ef_search"] == "250"
    # pgvector rejects hnsw.ef_search above 1000, so a larger limit is clamped.
    assert (await settings_after(5000))["ef_search"] == "1000"
    exact = await settings_after(10, exact=True)
    assert exact["indexscan"] == "off"
    assert exact["bitmapscan"] == "off"


@pg
async def test_dense_empty_temporal_scope_returns_nothing(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Refund window days."])

    before_everything = SearchScope(
        tenant_id, (collection_id,), as_of=datetime(2000, 1, 1, tzinfo=UTC)
    )

    assert await run_dense(db, tenant_id, before_everything, "refund window") == []
    assert await run_dense(db, tenant_id, SearchScope(tenant_id, ()), "refund window") == []


@pg
async def test_fts_or_semantics(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "The refund policy applies to every purchase.",
        "Window seats are reserved for members.",
        "Delivery takes several days.",
        "A refund request must arrive within the window of ten days.",
        "Harbor lanterns burn copper.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))

    hits = await run_fts(db, tenant_id, scope, "refund window days")

    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select id, text from passages where id = any(%s)", ([pid for pid, _ in hits],)
        )
        text_of = {row["id"]: row["text"] for row in await cur.fetchall()}
    returned = [text_of[passage_id] for passage_id, _ in hits]
    # Every passage with at least one of the three terms is returned; the unrelated one is not.
    assert sorted(returned) == sorted(texts[:4])
    # The passage with all three terms outranks those with one.
    assert returned[0] == texts[3]
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)
    assert isinstance(scores[0], float)


@pg
async def test_fts_stopwords_empty(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["The refund window is ten days."])
    scope = SearchScope(tenant_id, (collection_id,))

    assert await run_fts(db, tenant_id, scope, "the and of") == []
    assert await run_fts(db, tenant_id, scope, "") == []


@pg
async def test_fts_operators_in_query_are_plain_text(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = ["Refund requests are logged.", "The window closes at noon.", "Harbor lanterns."]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))

    hits = await run_fts(db, tenant_id, scope, "refund & !window | (days) :* <-> 'x")

    # Operator characters neither raise nor negate: refund and window both match.
    assert len(hits) == 2


@pg
async def test_fts_respects_scope_and_as_of(db, services):
    tenant_id, (x, y) = await make_scope(db, collections=2)
    (document_id,) = await seed_docs(
        db, services, tenant_id, x, ["Harbor copper lantern."], effective_from=D_JAN
    )
    await seed_docs(db, services, tenant_id, y, ["Harbor copper lantern elsewhere."])
    v2 = await add_version(db, services, tenant_id, document_id, "Harbor copper orchard.", D_JUN)
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select id from document_versions where version_number = 1 and document_id = %s",
            (document_id,),
        )
        v1_id = (await cur.fetchone())["id"]

    live = await run_fts(db, tenant_id, SearchScope(tenant_id, (x,)), "harbor copper")
    between = await run_fts(
        db,
        tenant_id,
        SearchScope(tenant_id, (x,), as_of=datetime(2026, 3, 1, tzinfo=UTC)),
        "harbor copper",
    )

    # Only X is readable: Y's passage never appears, and X shows one version per scope.
    assert await version_of(db, tenant_id, [pid for pid, _ in live]) == {v2.version_id}
    assert await version_of(db, tenant_id, [pid for pid, _ in between]) == {v1_id}
    assert len(live) == len(between) == 1
