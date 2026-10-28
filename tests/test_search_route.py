"""POST /api/v1/search, the worker-side `search` job, and its failure hook."""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from psycopg.types.json import Jsonb

from backend.config import Settings
from backend.jobs.queue import enqueue
from backend.jobs.registry import build_failure_handlers
from backend.services import Services
from tests.fakes import HashEmbedder, KeywordReranker
from tests.helpers.auth import csrf
from tests.pg_fixtures import drain, seed_docs
from tests.test_bm25 import D_JAN, D_JUL, add_version
from tests.test_collections import _create, _team
from tests.test_ingest_upload import context_for, jobs, json_logs

pytestmark = pytest.mark.pg

SEARCH = "/api/v1/search"
REFUND = "Refunds are issued within 30 days of purchase. Keep the receipt."
SHIPPING = "Shipping takes five business days for domestic orders."
WARRANTY = "The warranty covers manufacturing defects for two years."
CORPUS = [REFUND, SHIPPING, WARRANTY]
PASSAGE_HIT_KEYS = {
    "passage_id",
    "document_id",
    "version_id",
    "version_number",
    "effective_from",
    "document_title",
    "text",
    "start_char",
    "end_char",
    "page_start",
    "page_end",
    "section",
    "scores",
    "rank",
}


@pytest.fixture(autouse=True)
def _api_embedder(api_app):
    """The lifespan is skipped under ASGITransport, so the API's embedder is set here."""
    api_app.state.embedder = HashEmbedder()


@pytest.fixture
def short_wait(api_app, monkeypatch):
    monkeypatch.setattr(api_app.state.settings, "search_wait_s", 0.2)


class CountingReranker(KeywordReranker):
    def __init__(self) -> None:
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return super().score(query, passages)


class ExplodingReranker:
    model_id = "exploding"

    def score(self, query, passages):
        raise RuntimeError("SECRET-TEXT the reranker fell over")


class BrokenDb:
    def tenant_tx(self, tenant_id):
        raise ConnectionError("SECRET-TEXT database down")


@contextlib.asynccontextmanager
async def background_worker(db, services):
    """Drain the queue every 50 ms while the body runs, like a worker next to the API."""

    async def loop():
        while True:
            await drain(db, services)
            await asyncio.sleep(0.05)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def corpus(client, client2, client3, db, services, texts=CORPUS, **seed):
    """Alice owns a collection holding `texts`; Bob is a member without a grant."""
    alice, bob, carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    tenant_id = UUID(alice["tenant"]["id"])
    document_ids = await seed_docs(db, services, tenant_id, UUID(collection["id"]), texts, **seed)
    return alice, bob, carol, collection, tenant_id, document_ids


async def search(client, session, query="refunds receipt", **body):
    return await client.post(SEARCH, json={"query": query, **body}, headers=csrf(session))


async def count_jobs(db) -> int:
    async with db.system_tx() as conn:
        cur = await conn.execute("select count(*) as n from jobs")
        return (await cur.fetchone())["n"]


async def request_rows(db, tenant_id: UUID) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select * from search_requests order by created_at")
        return await cur.fetchall()


async def queue_search(db, session, collection_ids, query="refunds receipt", **params) -> UUID:
    """Insert a queued search request and its job, as the route does; return the request id."""
    tenant_id = UUID(session["tenant"]["id"])
    stored = {
        "k": 10,
        "mode": "hybrid",
        "collection_ids": [str(c) for c in collection_ids],
        "as_of": None,
        "known_at": None,
        **params,
    }
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into search_requests (tenant_id, user_id, query, params) "
            "values (%s, %s, %s, %s) returning id",
            (tenant_id, UUID(session["user"]["id"]), query, Jsonb(stored)),
        )
        request_id = (await cur.fetchone())["id"]
        await enqueue(
            conn, "search", {"search_request_id": str(request_id)}, tenant_id=tenant_id, priority=10
        )
    return request_id


async def insert_request(db, session, status, *, age_hours=0, error_code=None) -> UUID:
    """A request row in the given status whose `created_at` is `age_hours` ago."""
    tenant_id = UUID(session["tenant"]["id"])
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into search_requests "
            "(tenant_id, user_id, query, params, status, hits, error_code, created_at) "
            "values (%s, %s, 'q', %s, %s, %s, %s, now() - make_interval(hours => %s)) "
            "returning id",
            (
                tenant_id,
                UUID(session["user"]["id"]),
                Jsonb({}),
                status,
                Jsonb([]) if status == "done" else None,
                error_code,
                age_hours,
            ),
        )
        return (await cur.fetchone())["id"]


async def test_bm25_search_returns_hits_with_text(client, client2, client3, db, services):
    alice, *_ = await corpus(client, client2, client3, db, services)

    response = await search(client, alice, "refunds receipt", mode="bm25")

    body = response.json()
    assert response.status_code == 200, response.text
    assert set(body) == {"hits", "mode", "reranked", "took_ms"}
    assert (body["mode"], body["reranked"]) == ("bm25", False)
    assert isinstance(body["took_ms"], int)
    assert body["took_ms"] >= 0
    (hit,) = body["hits"]
    assert set(hit) == PASSAGE_HIT_KEYS
    assert hit["text"] == REFUND
    assert (hit["rank"], hit["version_number"], hit["document_title"]) == (1, 1, "Document 1")
    assert (hit["start_char"], hit["end_char"]) == (0, len(REFUND))
    assert list(hit["scores"]) == ["bm25"]


async def test_hybrid_without_rerank_runs_in_api(client, client2, client3, db, services):
    alice, _bob, _carol, _collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )

    # No worker is running: an unreranked search must not wait on one.
    response = await search(client, alice, "refunds receipt")

    body = response.json()
    assert response.status_code == 200, response.text
    assert (body["mode"], body["reranked"]) == ("hybrid", False)
    assert body["hits"][0]["text"] == REFUND
    assert "rrf" in body["hits"][0]["scores"]
    assert all("rerank" not in hit["scores"] for hit in body["hits"])
    assert await count_jobs(db) == 0
    assert await request_rows(db, tenant_id) == []


async def test_rerank_through_worker(client, client2, client3, db, services):
    alice, _bob, _carol, _collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )

    async with background_worker(db, services):
        response = await search(client, alice, "refunds receipt purchase", rerank=True)

    body = response.json()
    assert response.status_code == 200, response.text
    assert (body["mode"], body["reranked"]) == ("hybrid", True)
    assert all("rerank" in hit["scores"] for hit in body["hits"])
    assert body["hits"][0]["text"] == REFUND
    # KeywordReranker counts the distinct query tokens a passage contains.
    assert body["hits"][0]["scores"]["rerank"] == 3.0
    assert [hit["rank"] for hit in body["hits"]] == list(range(1, len(body["hits"]) + 1))
    assert await request_rows(db, tenant_id) == []
    (job,) = await jobs(db, "search")
    assert job["status"] == "succeeded"


async def test_rerank_request_row_holds_query_and_params(client, client2, client3, db, services):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )

    request = asyncio.create_task(
        search(client, alice, "refunds receipt", rerank=True, k=5, mode="bm25")
    )
    rows = []
    for _ in range(100):
        rows = await request_rows(db, tenant_id)
        if rows:
            break
        await asyncio.sleep(0.05)
    await drain(db, services)
    response = await asyncio.wait_for(request, 10)

    (row,) = rows
    assert row["query"] == "refunds receipt"
    assert row["status"] == "queued"
    assert row["user_id"] == UUID(alice["user"]["id"])
    assert row["params"] == {
        "k": 5,
        "mode": "bm25",
        "collection_ids": [collection["id"]],
        "as_of": None,
        "known_at": None,
    }
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "bm25"


async def test_rerank_timeout_504(client, client2, client3, db, services, short_wait):
    alice, *_ = await corpus(client, client2, client3, db, services)

    response = await search(client, alice, rerank=True)

    assert response.status_code == 504
    assert response.json()["code"] == "search_timeout"


async def test_timed_out_request_leaves_nothing_for_the_worker(
    client, client2, client3, db, services, short_wait
):
    alice, _bob, _carol, _collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    await search(client, alice, rerank=True)
    reranker = CountingReranker()

    # The job is still queued; by the time a worker claims it the caller is long gone.
    ran = await drain(db, Services(Settings(), db, HashEmbedder(), reranker))

    assert await request_rows(db, tenant_id) == []
    assert ran == 1
    assert reranker.calls == 0
    (job,) = await jobs(db, "search")
    assert job["status"] == "succeeded"


async def test_job_payload_has_only_id(client, client2, client3, db, services, short_wait):
    alice, _bob, _carol, _collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )

    await search(client, alice, "zebrafish quartz", rerank=True)

    (job,) = await jobs(db, "search")
    assert set(job["payload"]) == {"search_request_id"}
    UUID(job["payload"]["search_request_id"])
    assert job["tenant_id"] == tenant_id
    assert job["priority"] == 10
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select count(*) as n from jobs where jobs::text ilike '%zebrafish%'"
        )
        assert (await cur.fetchone())["n"] == 0


@pytest.mark.parametrize("rerank", [False, True])
async def test_unreadable_collection_404(client, client2, client3, db, services, rerank):
    _alice, bob, _carol, collection, _tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )

    hidden = await search(client2, bob, collection_ids=[collection["id"]], rerank=rerank)
    missing = await search(client2, bob, collection_ids=[str(uuid4())], rerank=rerank)

    for response in (hidden, missing):
        assert response.status_code == 404
        assert response.json()["code"] == "collection_not_found"
    assert await count_jobs(db) == 0


async def test_default_scope_is_the_readable_collections(client, client2, client3, db, services):
    alice, bob, *_ = await corpus(client, client2, client3, db, services)

    owner = await search(client, alice, mode="bm25")
    member = await search(client2, bob, mode="bm25")

    assert [hit["text"] for hit in owner.json()["hits"]] == [REFUND]
    assert member.status_code == 200
    assert member.json()["hits"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "a" * 2001},
        {"query": "a\u0000b"},
        {"query": "a\ud800b"},
        {"query": "ok", "k": 0},
        {"query": "ok", "k": 51},
        {"query": "ok", "mode": "semantic"},
        {"query": "ok", "collection_ids": ["not-a-uuid"]},
        {"query": "ok", "as_of": "2026-06-01T00:00:00"},
    ],
    ids=[
        "empty",
        "blank",
        "too_long",
        "nul",
        "surrogate",
        "k_zero",
        "k_over_max",
        "bad_mode",
        "bad_collection_id",
        "naive_as_of",
    ],
)
async def test_query_validation(client, client2, client3, db, services, body):
    alice, *_ = await corpus(client, client2, client3, db, services)

    response = await client.post(
        SEARCH,
        content=json.dumps(body),
        headers={**csrf(alice), "Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert await count_jobs(db) == 0


async def test_query_is_stripped_before_the_length_check(client, client2, client3, db, services):
    alice, *_ = await corpus(client, client2, client3, db, services)

    response = await search(client, alice, "  " + "a" * 2000 + "  ", mode="bm25")

    assert response.status_code == 200, response.text


async def test_search_requires_a_signed_in_caller(client):
    response = await client.post(SEARCH, json={"query": "refunds"})

    assert response.status_code == 401


async def test_as_of_search_returns_old_version(client, client2, client3, db, services):
    v1 = "Refunds are issued within 30 days."
    v2 = "Refunds are issued within 14 days and need a receipt."
    alice, _bob, _carol, _collection, tenant_id, (document_id,) = await corpus(
        client, client2, client3, db, services, [v1], effective_from=D_JAN
    )
    await add_version(db, services, tenant_id, document_id, v2, D_JUL)

    old = await search(client, alice, "refunds days", mode="bm25", as_of="2026-06-01T00:00:00Z")
    current = await search(client, alice, "refunds days", mode="bm25")
    before = await search(client, alice, "refunds days", mode="bm25", as_of="2025-12-31T00:00:00Z")

    assert [(h["text"], h["version_number"]) for h in old.json()["hits"]] == [(v1, 1)]
    assert [(h["text"], h["version_number"]) for h in current.json()["hits"]] == [(v2, 2)]
    assert before.status_code == 200
    assert before.json()["hits"] == []


async def test_rerank_honours_as_of(client, client2, client3, db, services):
    v1 = "Refunds are issued within 30 days."
    v2 = "Refunds are issued within 14 days and need a receipt."
    alice, _bob, _carol, _collection, tenant_id, (document_id,) = await corpus(
        client, client2, client3, db, services, [v1], effective_from=D_JAN
    )
    await add_version(db, services, tenant_id, document_id, v2, D_JUL)

    async with background_worker(db, services):
        old = await search(client, alice, "refunds days", rerank=True, as_of="2026-06-01T00:00:00Z")
        current = await search(client, alice, "refunds days", rerank=True)

    assert old.status_code == 200, old.text
    assert [(h["text"], h["version_number"]) for h in old.json()["hits"]] == [(v1, 1)]
    assert [(h["text"], h["version_number"]) for h in current.json()["hits"]] == [(v2, 2)]


async def test_queued_rerank_ignores_versions_created_after_the_request(
    client, client2, client3, db, services
):
    v1 = "Refunds are issued within 30 days."
    v2 = "Refunds are issued within 14 days and need a receipt."
    alice, _bob, _carol, _collection, tenant_id, (document_id,) = await corpus(
        client, client2, client3, db, services, [v1], effective_from=D_JAN
    )

    request = asyncio.create_task(
        search(client, alice, "refunds days", rerank=True, as_of="2026-06-01T00:00:00Z")
    )
    for _ in range(100):
        if await request_rows(db, tenant_id):
            break
        await asyncio.sleep(0.05)
    # Valid before the as-of date, but only recorded after the question was asked.
    await add_version(db, services, tenant_id, document_id, v2, datetime(2026, 3, 1, tzinfo=UTC))
    await drain(db, services)
    response = await asyncio.wait_for(request, 10)

    assert response.status_code == 200, response.text
    assert [(h["text"], h["version_number"]) for h in response.json()["hits"]] == [(v1, 1)]


async def test_job_stores_hits_and_marks_request_done(client, client2, client3, db, services):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    request_id = await queue_search(db, alice, [collection["id"]])

    assert await drain(db, services) == 1

    (row,) = await request_rows(db, tenant_id)
    assert row["id"] == request_id
    assert (row["status"], row["error_code"]) == ("done", None)
    assert row["completed_at"] is not None
    assert set(row["hits"][0]) == PASSAGE_HIT_KEYS
    assert row["hits"][0]["text"] == REFUND
    assert "rerank" in row["hits"][0]["scores"]
    (job,) = await jobs(db, "search")
    assert job["status"] == "succeeded"


async def test_job_failure_fails_the_request_without_retry(client, client2, client3, db, services):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    broken = Services(Settings(), db, HashEmbedder(), ExplodingReranker())
    await queue_search(db, alice, [collection["id"]], query="zebrafish quartz")

    with json_logs() as stream:
        assert await drain(db, broken) == 1

    (row,) = await request_rows(db, tenant_id)
    assert (row["status"], row["error_code"], row["hits"]) == ("failed", "search_failed", None)
    assert row["completed_at"] is not None
    (job,) = await jobs(db, "search")
    assert (job["status"], job["attempts"], job["last_error"]) == ("failed", 1, "search_failed")
    logs = stream.getvalue()
    assert "SECRET-TEXT" not in logs
    assert "zebrafish" not in logs


async def test_route_reports_502_when_the_job_fails(client, client2, client3, db, services):
    alice, _bob, _carol, _collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    broken = Services(Settings(), db, HashEmbedder(), ExplodingReranker())

    async with background_worker(db, broken):
        response = await search(client, alice, rerank=True)

    assert response.status_code == 502
    assert response.json()["code"] == "search_failed"
    assert await request_rows(db, tenant_id) == []


async def test_job_skips_a_request_that_is_not_queued(client, client2, client3, db, services):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    await queue_search(db, alice, [collection["id"]])
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute("update search_requests set status = 'failed', error_code = 'x'")
    reranker = CountingReranker()

    assert await drain(db, Services(Settings(), db, HashEmbedder(), reranker)) == 1

    (row,) = await request_rows(db, tenant_id)
    assert (row["status"], row["error_code"], row["hits"]) == ("failed", "x", None)
    assert reranker.calls == 0


async def test_job_deletes_old_requests_of_its_own_tenant_only(
    client, client2, client3, db, services
):
    alice, _bob, carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    old = await insert_request(db, alice, "done", age_hours=2)
    recent = await insert_request(db, alice, "done", age_hours=0)
    foreign_old = await insert_request(db, carol, "failed", age_hours=2, error_code="x")
    request_id = await queue_search(db, alice, [collection["id"]])

    assert await drain(db, services) == 1

    assert {row["id"] for row in await request_rows(db, tenant_id)} == {recent, request_id}
    assert old not in {row["id"] for row in await request_rows(db, tenant_id)}
    carol_tenant = UUID(carol["tenant"]["id"])
    assert [row["id"] for row in await request_rows(db, carol_tenant)] == [foreign_old]


async def context_of_search_job(db, services, session, collection_ids):
    request_id = await queue_search(db, session, collection_ids)
    (job,) = await jobs(db, "search")
    return request_id, job, context_for(job, db, services)


async def test_failure_hook_fails_a_queued_request(client, client2, client3, db, services):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    _id, _job, ctx = await context_of_search_job(db, services, alice, [collection["id"]])

    await build_failure_handlers(services)["search"](ctx, "lease_expired")

    (row,) = await request_rows(db, tenant_id)
    assert (row["status"], row["error_code"]) == ("failed", "lease_expired")
    assert row["completed_at"] is not None


@pytest.mark.parametrize(
    ("terminal", "code"), [("done", None), ("failed", "search_failed")], ids=["done", "failed"]
)
async def test_failure_hook_leaves_a_terminal_request_alone(
    client, client2, client3, db, services, terminal, code
):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    _id, _job, ctx = await context_of_search_job(db, services, alice, [collection["id"]])
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute(
            "update search_requests set status = %s, error_code = %s, hits = %s",
            (terminal, code, Jsonb([]) if terminal == "done" else None),
        )

    await build_failure_handlers(services)["search"](ctx, "lease_expired")

    (row,) = await request_rows(db, tenant_id)
    assert (row["status"], row["error_code"]) == (terminal, code)
    assert row["hits"] == ([] if terminal == "done" else None)


async def test_failure_hook_is_idempotent(client, client2, client3, db, services):
    alice, _bob, _carol, collection, tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    _id, _job, ctx = await context_of_search_job(db, services, alice, [collection["id"]])
    hook = build_failure_handlers(services)["search"]

    await hook(ctx, "lease_expired")
    await hook(ctx, "job_timeout")

    (row,) = await request_rows(db, tenant_id)
    assert (row["status"], row["error_code"]) == ("failed", "lease_expired")


@pytest.mark.parametrize("case", ["no_request_id", "no_tenant", "database_down", "bad_request_id"])
async def test_failure_hook_never_raises(client, client2, client3, db, services, case):
    alice, _bob, _carol, collection, _tenant_id, _docs = await corpus(
        client, client2, client3, db, services
    )
    await queue_search(db, alice, [collection["id"]])
    (job,) = await jobs(db, "search")
    job_row = dict(job)
    if case == "no_request_id":
        job_row["payload"] = {}
    elif case == "no_tenant":
        job_row["tenant_id"] = None
    elif case == "bad_request_id":
        job_row["payload"] = {"search_request_id": "not-a-uuid"}
    ctx = context_for(job_row, BrokenDb() if case == "database_down" else db, services)

    with json_logs() as stream:
        assert await build_failure_handlers(services)["search"](ctx, "internal_error") is None

    assert "SECRET-TEXT" not in stream.getvalue()
