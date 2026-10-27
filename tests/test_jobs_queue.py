import asyncio

import pytest
import pytest_asyncio

from backend.db.pool import Database
from backend.db.repos import tenants
from backend.jobs import queue

pytestmark = pytest.mark.pg

WORKER = "worker-a"
OTHER = "worker-b"


@pytest_asyncio.fixture(loop_scope="session")
async def pool5(db, pg_urls):
    """Five-connection pool; depends on `db` so its teardown empties the tables afterwards."""
    database = Database(pg_urls["app"], min_size=5, max_size=5)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


async def _row(db, job_id):
    async with db.system_tx() as conn:
        cur = await conn.execute("select * from jobs where id = %s", (job_id,))
        return await cur.fetchone()


async def _delta_seconds(db, job_id):
    """run_after minus the database's own now(), so Python clock skew cannot matter."""
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select extract(epoch from run_after - now()) as delta from jobs where id = %s",
            (job_id,),
        )
        return float((await cur.fetchone())["delta"])


async def _enqueue(db, kind="ingest", payload=None, **kwargs):
    kwargs.setdefault("tenant_id", None)
    async with db.system_tx() as conn:
        return await queue.enqueue(conn, kind, payload or {}, **kwargs)


async def _claim(db, worker=WORKER, kinds=("ingest",)):
    async with db.system_tx() as conn:
        return await queue.claim(conn, worker, kinds)


async def _sql(db, statement, params=()):
    async with db.system_tx() as conn:
        await conn.execute(statement, params)


async def _age_heartbeat(db, job_id, seconds):
    await _sql(
        db,
        "update jobs set heartbeat_at = now() - make_interval(secs => %s) where id = %s",
        (seconds, job_id),
    )


async def test_enqueue_in_rolled_back_tx_leaves_no_job(db):
    with pytest.raises(RuntimeError, match="boom"):
        async with db.system_tx() as conn:
            await queue.enqueue(conn, "ingest", {"doc": 1}, tenant_id=None)
            raise RuntimeError("boom")

    async with db.system_tx() as conn:
        assert await queue.count_queued(conn) == 0


async def test_enqueue_works_inside_tenant_tx(db):
    async with db.system_tx() as conn:
        tenant_id = (await tenants.insert_tenant(conn, "Acme"))["id"]

    async with db.tenant_tx(tenant_id) as conn:
        job_id = await queue.enqueue(conn, "ingest", {"doc": 7}, tenant_id=tenant_id)

    row = await _row(db, job_id)
    assert row["tenant_id"] == tenant_id
    assert row["payload"] == {"doc": 7}
    assert row["status"] == "queued"


async def test_enqueue_stores_options(db):
    job_id = await _enqueue(
        db, "embed", {"ids": [1, 2]}, priority=5, request_id="req-1", max_attempts=7
    )

    row = await _row(db, job_id)
    assert (row["kind"], row["priority"], row["request_id"], row["max_attempts"]) == (
        "embed",
        5,
        "req-1",
        7,
    )
    assert row["payload"] == {"ids": [1, 2]}


async def test_idempotency_key_returns_existing_id(db):
    first = await _enqueue(db, idempotency_key="k1")
    second = await _enqueue(db, idempotency_key="k1")
    other = await _enqueue(db, idempotency_key="k2")

    assert second == first
    assert other != first
    async with db.system_tx() as conn:
        assert await queue.count_queued(conn) == 2


async def test_concurrent_enqueue_with_same_key_returns_same_id(pool5):
    async def one():
        return await _enqueue(pool5, idempotency_key="shared")

    ids = await asyncio.gather(*(one() for _ in range(5)))

    assert len(set(ids)) == 1
    async with pool5.system_tx() as conn:
        assert await queue.count_queued(conn) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "x" * 101},
        {"a": {"b": 1}},
        {"a": [[1]]},
        {"a": [{"b": 1}]},
        {"a": list(range(1001))},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": b"bytes"},
        {1: "int key"},
    ],
)
async def test_payload_rejects_text_and_nesting(db, payload):
    with pytest.raises(ValueError):
        await _enqueue(db, payload=payload)

    async with db.system_tx() as conn:
        assert await queue.count_queued(conn) == 0


async def test_payload_error_names_key_not_value(db):
    secret = "s3cret-" + "x" * 100

    with pytest.raises(ValueError, match="document_text") as raised:
        await _enqueue(db, payload={"document_text": secret})

    assert secret not in str(raised.value)


async def test_payload_accepts_scalars_and_scalar_lists(db):
    payload = {
        "n": None,
        "b": True,
        "i": 3,
        "f": 1.5,
        "s": "x" * 100,
        "l": [1, "two", None, 4.5, False],
        "empty": [],
        "max": list(range(1000)),
    }

    job_id = await _enqueue(db, payload=payload)

    assert (await _row(db, job_id))["payload"] == payload


async def test_claim_priority_then_fifo(db):
    low = await _enqueue(db, priority=0)
    high_first = await _enqueue(db, priority=10)
    high_second = await _enqueue(db, priority=10)

    claimed = [(await _claim(db)).id for _ in range(3)]

    assert claimed == [high_first, high_second, low]
    assert await _claim(db) is None


async def test_claim_marks_running_and_returns_job(db):
    job_id = await _enqueue(
        db, "ingest", {"doc": 9}, priority=2, request_id="req-9", max_attempts=4
    )

    job = await _claim(db)

    assert job == queue.Job(
        id=job_id,
        tenant_id=None,
        kind="ingest",
        payload={"doc": 9},
        priority=2,
        attempts=1,
        max_attempts=4,
        request_id="req-9",
    )
    row = await _row(db, job_id)
    assert row["status"] == "running"
    assert row["locked_by"] == WORKER
    assert row["locked_at"] is not None
    assert row["heartbeat_at"] is not None


async def test_claim_respects_run_after_and_kinds(db):
    future = await _enqueue(db, "ingest")
    await _sql(db, "update jobs set run_after = now() + interval '1 hour' where id = %s", (future,))
    other_kind = await _enqueue(db, "embed")
    ready = await _enqueue(db, "ingest")

    assert (await _claim(db, kinds=["ingest"])).id == ready
    assert await _claim(db, kinds=["ingest"]) is None
    assert (await _claim(db, kinds=["embed", "ingest"])).id == other_kind
    assert await _claim(db, kinds=[]) is None


async def test_enqueue_run_after_delays_claim(db):
    async with db.system_tx() as conn:
        cur = await conn.execute("select now() + interval '1 hour' as later")
        later = (await cur.fetchone())["later"]
    job_id = await _enqueue(db, run_after=later)

    assert await _claim(db) is None
    assert 3500 <= await _delta_seconds(db, job_id) <= 3600


async def test_concurrent_claims_never_duplicate(pool5):
    for _ in range(20):
        await _enqueue(pool5)

    async def claimer(name):
        claimed = []
        while True:
            async with pool5.system_tx() as conn:
                job = await queue.claim(conn, name, ["ingest"])
            if job is None:
                return claimed
            claimed.append(job.id)

    results = await asyncio.gather(*(claimer(f"w{i}") for i in range(5)))

    all_ids = [job_id for ids in results for job_id in ids]
    assert len(all_ids) == 20
    assert len(set(all_ids)) == 20


async def test_retry_backoff_then_dead(db):
    job_id = await _enqueue(db, max_attempts=3)

    await _claim(db)
    async with db.system_tx() as conn:
        assert await queue.fail(conn, job_id, WORKER, "model_timeout") == "queued"
    row = await _row(db, job_id)
    assert row["status"] == "queued"
    assert row["last_error"] == "model_timeout"
    assert row["locked_by"] is None
    assert 4 <= await _delta_seconds(db, job_id) <= 6

    await _sql(db, "update jobs set run_after = now() where id = %s", (job_id,))
    assert (await _claim(db)).attempts == 2
    async with db.system_tx() as conn:
        assert await queue.fail(conn, job_id, WORKER, "model_timeout") == "queued"
    assert 19 <= await _delta_seconds(db, job_id) <= 21

    await _sql(db, "update jobs set run_after = now() where id = %s", (job_id,))
    assert (await _claim(db)).attempts == 3
    async with db.system_tx() as conn:
        assert await queue.fail(conn, job_id, WORKER, "model_timeout") == "dead"
    row = await _row(db, job_id)
    assert row["status"] == "dead"
    assert row["last_error"] == "model_timeout"
    assert await _claim(db) is None


async def test_permanent_failure(db):
    job_id = await _enqueue(db)
    await _claim(db)

    async with db.system_tx() as conn:
        outcome = await queue.fail(conn, job_id, WORKER, "bad_file", permanent=True)

    assert outcome == "failed"
    row = await _row(db, job_id)
    assert row["status"] == "failed"
    assert row["last_error"] == "bad_file"
    assert row["locked_by"] is None


@pytest.mark.parametrize(
    "code", ["Has Spaces", "", "UPPER", "a" * 65, "semi;colon", "dash-ed", "ok\n"]
)
async def test_invalid_error_code_rejected(db, code):
    job_id = await _enqueue(db)
    await _claim(db)

    with pytest.raises(ValueError):
        async with db.system_tx() as conn:
            await queue.fail(conn, job_id, WORKER, code)

    assert (await _row(db, job_id))["status"] == "running"


async def test_error_code_at_length_limit_accepted(db):
    job_id = await _enqueue(db)
    await _claim(db)

    async with db.system_tx() as conn:
        assert await queue.fail(conn, job_id, WORKER, "a" * 64, permanent=True) == "failed"


async def test_complete_marks_succeeded(db):
    job_id = await _enqueue(db)
    await _claim(db)

    async with db.system_tx() as conn:
        assert await queue.complete(conn, job_id, WORKER) is True
        assert await queue.complete(conn, job_id, WORKER) is False

    row = await _row(db, job_id)
    assert row["status"] == "succeeded"
    assert row["locked_by"] is None


async def test_heartbeat_wrong_worker_false(db):
    job_id = await _enqueue(db)
    await _claim(db)
    await _age_heartbeat(db, job_id, 60)

    async with db.system_tx() as conn:
        assert await queue.heartbeat(conn, job_id, OTHER) is False
    stale = (await _row(db, job_id))["heartbeat_at"]
    async with db.system_tx() as conn:
        assert await queue.heartbeat(conn, job_id, WORKER) is True
    fresh = (await _row(db, job_id))["heartbeat_at"]

    assert fresh > stale


async def test_wrong_worker_cannot_complete_fail_or_release(db):
    job_id = await _enqueue(db)
    await _claim(db)

    async with db.system_tx() as conn:
        assert await queue.complete(conn, job_id, OTHER) is False
        assert await queue.fail(conn, job_id, OTHER, "bad_file") is None
        assert await queue.release(conn, job_id, OTHER) is False

    row = await _row(db, job_id)
    assert (row["status"], row["locked_by"], row["attempts"]) == ("running", WORKER, 1)


async def test_mutations_on_non_running_job_rejected(db):
    job_id = await _enqueue(db)

    async with db.system_tx() as conn:
        assert await queue.heartbeat(conn, job_id, WORKER) is False
        assert await queue.complete(conn, job_id, WORKER) is False
        assert await queue.fail(conn, job_id, WORKER, "bad_file") is None
        assert await queue.release(conn, job_id, WORKER) is False

    assert (await _row(db, job_id))["status"] == "queued"


async def test_release_restores_attempt(db):
    job_id = await _enqueue(db)
    claimed = await _claim(db)
    assert claimed.attempts == 1

    async with db.system_tx() as conn:
        assert await queue.release(conn, job_id, WORKER) is True

    row = await _row(db, job_id)
    assert (row["status"], row["attempts"], row["locked_by"], row["locked_at"]) == (
        "queued",
        0,
        None,
        None,
    )
    assert (await _claim(db)).attempts == 1


async def test_requeue_stale_counts_attempt(db):
    job_id = await _enqueue(db, max_attempts=3)
    await _claim(db)
    await _age_heartbeat(db, job_id, 121)

    async with db.system_tx() as conn:
        assert await queue.requeue_stale(conn) == 1

    row = await _row(db, job_id)
    assert (row["status"], row["attempts"], row["locked_by"]) == ("queued", 1, None)
    assert await _delta_seconds(db, job_id) <= 1
    assert (await _claim(db, worker=OTHER)).attempts == 2


async def test_requeue_stale_exhausted_goes_dead(db):
    job_id = await _enqueue(db, max_attempts=1)
    await _claim(db)
    await _age_heartbeat(db, job_id, 121)

    async with db.system_tx() as conn:
        assert await queue.requeue_stale(conn) == 1

    row = await _row(db, job_id)
    assert (row["status"], row["last_error"], row["locked_by"]) == ("dead", "lease_expired", None)
    assert await _claim(db) is None


async def test_requeue_stale_leaves_fresh_leases_alone(db):
    running = await _enqueue(db)
    await _claim(db)
    await _age_heartbeat(db, running, 119)
    queued = await _enqueue(db)

    async with db.system_tx() as conn:
        assert await queue.requeue_stale(conn) == 0
    assert (await _row(db, running))["status"] == "running"

    async with db.system_tx() as conn:
        assert await queue.requeue_stale(conn, stale_after_s=60.0) == 1
    assert (await _row(db, running))["status"] == "queued"
    assert (await _row(db, queued))["status"] == "queued"


async def test_count_queued(db):
    async with db.system_tx() as conn:
        assert await queue.count_queued(conn) == 0

    for _ in range(3):
        await _enqueue(db)
    claimed = await _claim(db)

    async with db.system_tx() as conn:
        assert await queue.count_queued(conn) == 2
        await queue.complete(conn, claimed.id, WORKER)
        assert await queue.count_queued(conn) == 2


@pytest.mark.parametrize("code", ["a", "internal_error", "job_timeout", "x1_" * 20, "0" * 64])
def test_is_valid_error_code_accepts_lowercase_codes(code):
    assert queue.is_valid_error_code(code) is True


@pytest.mark.parametrize(
    "code", ["", "Bad-Input", "has space", "UPPER", "x" * 65, "line\nbreak", None]
)
def test_is_valid_error_code_rejects_everything_else(code):
    assert queue.is_valid_error_code(code) is False
