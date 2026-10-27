import asyncio
import io
import json
import logging
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from backend.cli.main import _run_worker, app
from backend.config import Settings
from backend.jobs import queue
from backend.jobs.registry import build_handlers
from backend.jobs.worker import (
    JobContext,
    PermanentJobError,
    RetryableJobError,
    Worker,
)
from backend.obs.logging import JsonFormatter
from backend.obs.metrics import JOB_DURATION, JOBS

LIMIT_S = 10.0


async def _enqueue(db, kind="ingest", payload=None, **kwargs):
    kwargs.setdefault("tenant_id", None)
    async with db.system_tx() as conn:
        return await queue.enqueue(conn, kind, payload or {}, **kwargs)


async def _job(db, job_id):
    async with db.system_tx() as conn:
        cur = await conn.execute("select * from jobs where id = %s", (job_id,))
        return await cur.fetchone()


async def _worker_rows(db):
    async with db.system_tx() as conn:
        cur = await conn.execute("select id, kinds from workers")
        return await cur.fetchall()


async def _until(predicate, attempts=500):
    """Poll an async predicate for about 10 s; a condition that never holds fails fast."""
    for _ in range(attempts):
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


async def _run_once(worker):
    return await asyncio.wait_for(worker.run_once(), LIMIT_S)


@contextmanager
def _json_logs():
    """Capture log output through the production formatter, which never prints messages."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


async def _noop(ctx: JobContext):
    return None


def test_build_handlers_is_empty_in_plan_a():
    assert build_handlers(None) == {}


def test_worker_help_exits_zero():
    result = CliRunner().invoke(app, ["worker", "--help"])
    assert result.exit_code == 0


@pytest.mark.pg
async def test_run_once_success(db):
    seen = []

    async def handler(ctx: JobContext):
        seen.append(ctx.job.payload)

    job_id = await _enqueue(db, payload={"doc": 7})
    worker = Worker(db, {"ingest": handler})

    assert await _run_once(worker) is True
    assert seen == [{"doc": 7}]
    assert (await _job(db, job_id))["status"] == "succeeded"
    assert await _run_once(worker) is False


@pytest.mark.pg
async def test_handler_receives_context(db):
    seen = {}

    async def handler(ctx: JobContext):
        seen.update(job_id=ctx.job.id, db=ctx.db, worker_id=ctx.worker_id, services=ctx.services)

    job_id = await _enqueue(db)
    sentinel = object()
    worker = Worker(db, {"ingest": handler}, services=sentinel, worker_id="w-ctx")

    assert await _run_once(worker) is True
    assert seen == {"job_id": job_id, "db": db, "worker_id": "w-ctx", "services": sentinel}


@pytest.mark.pg
async def test_permanent_error_marks_failed(db):
    async def handler(ctx: JobContext):
        raise PermanentJobError("bad_input")

    job_id = await _enqueue(db)
    await _run_once(Worker(db, {"ingest": handler}))

    row = await _job(db, job_id)
    assert row["status"] == "failed"
    assert row["last_error"] == "bad_input"


@pytest.mark.pg
async def test_retryable_error_requeues_with_its_code(db):
    async def handler(ctx: JobContext):
        raise RetryableJobError("upstream_down")

    job_id = await _enqueue(db)
    await _run_once(Worker(db, {"ingest": handler}))

    row = await _job(db, job_id)
    assert row["status"] == "queued"
    assert row["last_error"] == "upstream_down"


@pytest.mark.pg
async def test_retryable_error_on_last_attempt_is_dead(db):
    async def handler(ctx: JobContext):
        raise RetryableJobError("upstream_down")

    job_id = await _enqueue(db, max_attempts=1)
    await _run_once(Worker(db, {"ingest": handler}))

    assert (await _job(db, job_id))["status"] == "dead"


@pytest.mark.pg
async def test_unexpected_error_retries_without_leaking_message(db):
    async def handler(ctx: JobContext):
        raise ValueError("SECRET-DOC")

    job_id = await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}))

    row = await _job(db, job_id)
    assert row["status"] == "queued"
    assert row["last_error"] == "internal_error"
    logs = stream.getvalue()
    assert "SECRET-DOC" not in logs
    assert "job handler error" in logs
    assert "ValueError" in logs


@pytest.mark.pg
async def test_timeout_retries(db):
    async def handler(ctx: JobContext):
        await asyncio.sleep(1)

    job_id = await _enqueue(db, kind="slow")
    worker = Worker(db, {"slow": handler}, timeouts_s={"slow": 0.05})
    await _run_once(worker)

    row = await _job(db, job_id)
    assert row["status"] == "queued"
    assert row["last_error"] == "job_timeout"


@pytest.mark.pg
async def test_default_timeout_applies_to_kinds_without_override(db):
    async def handler(ctx: JobContext):
        await asyncio.sleep(1)

    job_id = await _enqueue(db)
    await _run_once(Worker(db, {"ingest": handler}, default_timeout_s=0.05))

    assert (await _job(db, job_id))["last_error"] == "job_timeout"


@pytest.mark.pg
async def test_heartbeat_advances_during_job(db):
    seen = {}

    async def handler(ctx: JobContext):
        await asyncio.sleep(0.3)
        async with ctx.db.system_tx() as conn:
            cur = await conn.execute(
                "select heartbeat_at > locked_at as advanced from jobs where id = %s",
                (ctx.job.id,),
            )
            seen["advanced"] = (await cur.fetchone())["advanced"]

    await _enqueue(db)
    await _run_once(Worker(db, {"ingest": handler}, heartbeat_s=0.05))

    assert seen == {"advanced": True}


@pytest.mark.pg
async def test_unregistered_kind_not_claimed(db):
    async def handler(ctx: JobContext):
        raise AssertionError("must not run")

    other = await _enqueue(db, kind="other")
    worker = Worker(db, {"ingest": handler})

    assert await _run_once(worker) is False
    row = await _job(db, other)
    assert row["status"] == "queued"
    assert row["attempts"] == 0


@pytest.mark.pg
async def test_graceful_stop_releases_job(db):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(ctx: JobContext):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    job_id = await _enqueue(db)
    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.05, shutdown_grace_s=0.1)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.wait_for(started.wait(), LIMIT_S)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert cancelled.is_set()
    row = await _job(db, job_id)
    assert row["status"] == "queued"
    assert row["attempts"] == 0
    assert row["locked_by"] is None


@pytest.mark.pg
async def test_stop_lets_a_job_finish_within_the_grace_period(db):
    started = asyncio.Event()

    async def handler(ctx: JobContext):
        started.set()
        await asyncio.sleep(0.2)

    job_id = await _enqueue(db)
    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.05, shutdown_grace_s=5.0)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.wait_for(started.wait(), LIMIT_S)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert (await _job(db, job_id))["status"] == "succeeded"


@pytest.mark.pg
async def test_idle_worker_wakes_immediately_on_stop(db):
    worker = Worker(db, {"ingest": _noop}, poll_interval_s=30.0)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.sleep(0.2)
        stop.set()
        await asyncio.wait_for(task, 5.0)
    finally:
        task.cancel()


@pytest.mark.pg
async def test_run_processes_jobs_enqueued_while_idle(db):
    done = asyncio.Event()

    async def handler(ctx: JobContext):
        done.set()

    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.05)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.sleep(0.1)
        job_id = await _enqueue(db)
        await asyncio.wait_for(done.wait(), LIMIT_S)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert (await _job(db, job_id))["status"] == "succeeded"


@pytest.mark.pg
async def test_worker_row_lifecycle(db):
    worker = Worker(db, {"b": _noop, "a": _noop}, worker_id="w-life")
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:

        async def row_exists():
            return bool(await _worker_rows(db))

        await _until(row_exists)
        rows = await _worker_rows(db)
        assert [(r["id"], sorted(r["kinds"])) for r in rows] == [("w-life", ["a", "b"])]
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert await _worker_rows(db) == []


@pytest.mark.pg
async def test_run_requeues_stale_jobs_of_dead_workers(db):
    done = asyncio.Event()

    async def handler(ctx: JobContext):
        done.set()

    job_id = await _enqueue(db)
    async with db.system_tx() as conn:
        claimed = await queue.claim(conn, "dead-worker", ["ingest"])
        await conn.execute(
            "update jobs set heartbeat_at = now() - interval '1 hour' where id = %s", (job_id,)
        )
    assert claimed is not None

    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.05, stale_after_s=60.0)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.wait_for(done.wait(), LIMIT_S)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert (await _job(db, job_id))["status"] == "succeeded"


@pytest.mark.pg
async def test_jobs_metric_incremented(db):
    async def bad(ctx: JobContext):
        raise PermanentJobError("bad_input")

    async def flaky(ctx: JobContext):
        raise RetryableJobError("upstream_down")

    def count(kind, outcome):
        return JOBS.labels(kind=kind, outcome=outcome).value

    duration_before = JOB_DURATION.labels(kind="m_ok").total
    before = {
        ("m_ok", "succeeded"): count("m_ok", "succeeded"),
        ("m_bad", "failed"): count("m_bad", "failed"),
        ("m_retry", "retry"): count("m_retry", "retry"),
        ("m_dead", "dead"): count("m_dead", "dead"),
    }
    await _enqueue(db, kind="m_ok")
    await _enqueue(db, kind="m_bad")
    await _enqueue(db, kind="m_retry")
    await _enqueue(db, kind="m_dead", max_attempts=1)
    worker = Worker(db, {"m_ok": _noop, "m_bad": bad, "m_retry": flaky, "m_dead": flaky})
    for _ in range(4):
        assert await _run_once(worker) is True

    for (kind, outcome), value in before.items():
        assert count(kind, outcome) == value + 1, (kind, outcome)
    assert JOB_DURATION.labels(kind="m_ok").total == duration_before + 1


@pytest.mark.pg
async def test_log_lines_carry_job_context(db):
    job_id = await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": _noop}, worker_id="w-log"))

    lines = [line for line in stream.getvalue().splitlines() if '"outcome"' in line]
    assert len(lines) == 1
    assert f'"job_id":"{job_id}"' in lines[0]
    assert '"kind":"ingest"' in lines[0]
    assert '"attempt":1' in lines[0]
    assert '"outcome":"succeeded"' in lines[0]
    assert '"worker_id":"w-log"' in lines[0]


@pytest.mark.pg
@pytest.mark.parametrize("code", ["Bad-Input", "", "x" * 65, "with space"])
async def test_permanent_error_with_invalid_code_is_failed_as_internal_error(db, code):
    async def handler(ctx: JobContext):
        raise PermanentJobError(code)

    before = JOBS.labels(kind="ingest", outcome="failed").value
    job_id = await _enqueue(db)

    assert await _run_once(Worker(db, {"ingest": handler})) is True

    row = await _job(db, job_id)
    assert row["status"] == "failed"
    assert row["last_error"] == "internal_error"
    assert row["locked_by"] is None
    assert JOBS.labels(kind="ingest", outcome="failed").value == before + 1


@pytest.mark.pg
async def test_retryable_error_with_invalid_code_is_retried_as_internal_error(db):
    async def handler(ctx: JobContext):
        raise RetryableJobError("Bad-Input")

    job_id = await _enqueue(db)
    await _run_once(Worker(db, {"ingest": handler}))

    row = await _job(db, job_id)
    assert row["status"] == "queued"
    assert row["last_error"] == "internal_error"
    assert row["locked_by"] is None


@pytest.mark.pg
async def test_run_does_not_leave_a_job_running_after_an_invalid_code(db):
    async def handler(ctx: JobContext):
        raise PermanentJobError("Bad-Input")

    job_id = await _enqueue(db)
    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.05)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:

        async def failed():
            return (await _job(db, job_id))["status"] == "failed"

        await _until(failed)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert (await _job(db, job_id))["last_error"] == "internal_error"


@pytest.mark.pg
async def test_handler_cancelled_without_shutdown_is_an_internal_error_retry(db):
    async def handler(ctx: JobContext):
        raise asyncio.CancelledError

    job_id = await _enqueue(db)
    with _json_logs() as stream:
        assert await _run_once(Worker(db, {"ingest": handler})) is True

    row = await _job(db, job_id)
    assert row["status"] == "queued"
    assert row["attempts"] == 1
    assert row["last_error"] == "internal_error"
    assert "job released" not in stream.getvalue()


@pytest.mark.pg
async def test_run_does_not_hot_loop_on_a_self_cancelling_handler(db):
    calls = []

    async def handler(ctx: JobContext):
        calls.append(1)
        raise asyncio.CancelledError

    job_id = await _enqueue(db)
    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.02)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.sleep(0.4)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    row = await _job(db, job_id)
    assert len(calls) == 1
    assert (row["status"], row["attempts"]) == ("queued", 1)


@pytest.mark.pg
async def test_handler_that_ignores_cancellation_is_abandoned_not_released(db):
    started = asyncio.Event()
    let_go = asyncio.Event()

    async def handler(ctx: JobContext):
        started.set()
        while not let_go.is_set():
            try:
                await asyncio.wait_for(let_go.wait(), 30)
            except asyncio.CancelledError:
                continue  # deliberately swallow cancellation

    job_id = await _enqueue(db)
    worker = Worker(
        db, {"ingest": handler}, worker_id="w-stuck", poll_interval_s=0.05, shutdown_grace_s=0.1
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        with _json_logs() as stream:
            await asyncio.wait_for(started.wait(), LIMIT_S)
            stop.set()
            await asyncio.wait_for(task, 3.0)
        row = await _job(db, job_id)
        assert (row["status"], row["locked_by"]) == ("running", "w-stuck")
        logs = stream.getvalue()
        assert "job handler did not stop" in logs
        assert "job released" not in logs
    finally:
        let_go.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.pg
async def test_timeout_raised_inside_the_handler_is_an_internal_error(db):
    async def handler(ctx: JobContext):
        raise TimeoutError("SECRET-DOC")

    job_id = await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}))

    assert (await _job(db, job_id))["last_error"] == "internal_error"
    logs = stream.getvalue()
    assert "job handler error" in logs
    assert "TimeoutError" in logs
    assert "SECRET-DOC" not in logs


@pytest.mark.pg
async def test_handler_error_log_line_carries_job_context(db):
    async def handler(ctx: JobContext):
        raise ValueError("SECRET-DOC")

    job_id = await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}, worker_id="w-err"))

    lines = [line for line in stream.getvalue().splitlines() if "job handler error" in line]
    assert len(lines) == 1
    assert f'"job_id":"{job_id}"' in lines[0]
    assert '"kind":"ingest"' in lines[0]
    assert '"attempt":1' in lines[0]
    assert '"worker_id":"w-err"' in lines[0]


@pytest.mark.pg
async def test_outcome_line_carries_error_code_for_failures(db):
    async def handler(ctx: JobContext):
        raise RetryableJobError("upstream_down")

    await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}))

    lines = [line for line in stream.getvalue().splitlines() if '"outcome"' in line]
    assert len(lines) == 1
    assert '"outcome":"retry"' in lines[0]
    assert '"error_code":"upstream_down"' in lines[0]


@pytest.mark.pg
async def test_lease_lost_log_line_carries_job_context(db):
    async def handler(ctx: JobContext):
        async with ctx.db.system_tx() as conn:
            await conn.execute("update jobs set locked_by = 'thief' where id = %s", (ctx.job.id,))

    job_id = await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}, worker_id="w-lost"))

    lines = [line for line in stream.getvalue().splitlines() if "job lease lost" in line]
    assert len(lines) == 1
    assert f'"job_id":"{job_id}"' in lines[0]
    assert '"kind":"ingest"' in lines[0]
    assert '"attempt":1' in lines[0]


@pytest.mark.pg
async def test_heartbeat_lease_lost_line_carries_job_context(db):
    async def handler(ctx: JobContext):
        async with ctx.db.system_tx() as conn:
            await conn.execute("update jobs set locked_by = 'thief' where id = %s", (ctx.job.id,))
        await asyncio.sleep(0.3)

    job_id = await _enqueue(db)
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}, heartbeat_s=0.05))

    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    lost = [entry for entry in lines if entry["msg"] == "job lease lost"]
    assert len(lost) >= 2, "expected the heartbeat line and the completion line"
    for entry in lost:
        assert entry["job_id"] == str(job_id)
        assert entry["kind"] == "ingest"
        assert entry["attempt"] == 1


@pytest.mark.pg
async def test_released_is_not_logged_when_the_lease_was_already_lost(db):
    started = asyncio.Event()

    async def handler(ctx: JobContext):
        async with ctx.db.system_tx() as conn:
            await conn.execute("update jobs set locked_by = 'thief' where id = %s", (ctx.job.id,))
        started.set()
        await asyncio.Event().wait()

    await _enqueue(db)
    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.05, shutdown_grace_s=0.1)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        with _json_logs() as stream:
            await asyncio.wait_for(started.wait(), LIMIT_S)
            stop.set()
            await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    logs = stream.getvalue()
    assert "job released" not in logs
    assert "job lease lost" in logs


@pytest.mark.pg
async def test_run_survives_a_failing_queue_call(db, monkeypatch):
    real_claim = queue.claim
    calls = []

    async def flaky_claim(conn, worker_id, kinds):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return await real_claim(conn, worker_id, kinds)

    monkeypatch.setattr(queue, "claim", flaky_claim)
    done = asyncio.Event()

    async def handler(ctx: JobContext):
        done.set()

    job_id = await _enqueue(db)
    worker = Worker(db, {"ingest": handler}, poll_interval_s=0.02)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.wait_for(done.wait(), LIMIT_S)
        stop.set()
        await asyncio.wait_for(task, LIMIT_S)
    finally:
        task.cancel()

    assert len(calls) >= 2
    assert (await _job(db, job_id))["status"] == "succeeded"


@pytest.mark.pg
async def test_request_id_is_bound_for_the_duration_of_the_job(db):
    async def handler(ctx: JobContext):
        raise ValueError("SECRET-DOC")

    await _enqueue(db, request_id="req-1")
    with _json_logs() as stream:
        await _run_once(Worker(db, {"ingest": handler}))

    lines = [line for line in stream.getvalue().splitlines() if "job_id" in line]
    assert {"job handler error", "job finished"} <= {json.loads(line)["msg"] for line in lines}
    assert all(json.loads(line)["request_id"] == "req-1" for line in lines)


@pytest.mark.pg
async def test_request_id_does_not_leak_to_the_next_job(db):
    async def handler(ctx: JobContext):
        return None

    await _enqueue(db, request_id="req-1")
    await _enqueue(db)
    worker = Worker(db, {"ingest": handler})
    with _json_logs() as stream:
        await _run_once(worker)
        await _run_once(worker)

    finished = [json.loads(line) for line in stream.getvalue().splitlines()]
    finished = [entry for entry in finished if entry["msg"] == "job finished"]
    assert [entry.get("request_id") for entry in finished] == ["req-1", None]


def test_worker_command_rejects_concurrency_below_one(monkeypatch):
    started = []

    async def spy(settings):
        started.append(settings)

    monkeypatch.setattr("backend.cli.main._run_worker", spy)
    monkeypatch.setenv("HALLUCISCOPE_WORKER_CONCURRENCY", "0")

    result = CliRunner().invoke(app, ["worker"])

    assert result.exit_code == 2
    assert "worker_concurrency" in result.output
    assert started == []


class _Harness:
    """Fakes for everything _run_worker opens, recording the order of events."""

    def __init__(self, monkeypatch, *, slot_zero_raises=False, metrics_raises=False):
        self.events = []
        events = self.events

        class FakeDb:
            def __init__(self, url, **kwargs):
                pass

            async def open(self):
                events.append("db.open")

            async def close(self):
                events.append("db.close")

        class FakeServer:
            def shutdown(self):
                events.append("metrics.shutdown")

            def server_close(self):
                events.append("metrics.close")

        def start_metrics(host, port, token):
            if metrics_raises:
                raise OSError("port in use")
            events.append("metrics.start")
            return FakeServer()

        class FakeWorker:
            def __init__(self, db, handlers, *, services=None, worker_id):
                self.worker_id = worker_id

            async def run(self, stop):
                if slot_zero_raises and self.worker_id.endswith(":0"):
                    raise RuntimeError("boom")
                await stop.wait()
                await asyncio.sleep(0.05)
                events.append(f"slot{self.worker_id[-1]}.stopped")

        monkeypatch.setattr("backend.db.pool.Database", FakeDb)
        monkeypatch.setattr("backend.jobs.worker.Worker", FakeWorker)
        monkeypatch.setattr("backend.obs.metrics_server.start_metrics_server", start_metrics)
        monkeypatch.setattr("backend.obs.logging.configure_logging", lambda level: None)


async def test_run_worker_waits_for_other_slots_when_one_slot_fails(monkeypatch):
    harness = _Harness(monkeypatch, slot_zero_raises=True)

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(_run_worker(Settings(worker_concurrency=2)), LIMIT_S)

    events = harness.events
    assert "slot1.stopped" in events
    assert events.index("slot1.stopped") < events.index("metrics.shutdown")
    assert events.index("metrics.close") < events.index("db.close")
    assert events[-1] == "db.close"


async def test_run_worker_closes_the_database_when_the_metrics_server_fails(monkeypatch):
    harness = _Harness(monkeypatch, metrics_raises=True)

    with pytest.raises(OSError):
        await asyncio.wait_for(_run_worker(Settings(worker_concurrency=1)), LIMIT_S)

    assert harness.events == ["db.open", "db.close"]
