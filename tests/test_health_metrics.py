import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import psycopg
import pytest

from backend.db.migrate import Migration, applied_versions, discover, migrate
from backend.jobs import queue

pytestmark = pytest.mark.pg

LIVE = "/api/v1/health/live"
READY = "/api/v1/health/ready"
METRICS = "/api/v1/metrics"
TOKEN = "metrics-secret"


def _llm_ok(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/version"
    return httpx.Response(200, json={"version": "0.0.0"})


def _llm_down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


@pytest.fixture
async def llm(api_app):
    """Install an app.state.http whose transport the test controls."""
    clients = []

    def install(handler):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(http)
        api_app.state.http = http

    yield install
    for http in clients:
        await http.aclose()


async def _sql(db, statement, params=()):
    async with db.system_tx() as conn:
        await conn.execute(statement, params)


async def test_live(client):
    response = await client.get(LIVE)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_all_ok(client, db, llm):
    await _sql(db, "insert into workers (id, kinds) values ('w1', array['ingest'])")
    llm(_llm_ok)

    response = await client.get(READY)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "db": "ok",
        "migrations": "ok",
        "worker": "ok",
        "llm": "ok",
    }


async def test_ready_reports_stale_worker_and_llm_down(client, llm):
    llm(_llm_down)

    response = await client.get(READY)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["worker"] == "stale"
    assert body["llm"] == "unavailable"


async def test_ready_old_heartbeat_is_stale(client, db, llm):
    await _sql(
        db,
        "insert into workers (id, kinds, heartbeat_at) "
        "values ('w1', array['ingest'], now() - interval '61 seconds')",
    )
    llm(_llm_ok)

    response = await client.get(READY)

    assert response.json()["worker"] == "stale"


async def test_ready_pending_migrations_503(client, llm, monkeypatch):
    llm(_llm_ok)
    from backend.api.routes import health

    def with_future_migration(*args, **kwargs):
        future = Migration(9999, "future", None, "0" * 64)
        return [*discover(*args, **kwargs), future]

    monkeypatch.setattr(health, "discover", with_future_migration)

    response = await client.get(READY)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["db"] == "ok"
    assert body["migrations"] == "pending"


async def test_ready_db_failure_503_without_error_text(client, api_app, llm, caplog):
    llm(_llm_ok)

    class BrokenDb:
        def system_tx(self):
            raise RuntimeError("secret-host-password")

    api_app.state.db = BrokenDb()

    with caplog.at_level(logging.ERROR):
        response = await client.get(READY)

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["db"] == "error"
    assert "secret-host-password" not in response.text
    assert any(
        r.getMessage() == "readiness check failed" and r.stage == "db" for r in caplog.records
    )


class _HangingDb:
    """A database whose transactions never complete."""

    def system_tx(self):
        @asynccontextmanager
        async def hang():
            await asyncio.Event().wait()
            yield

        return hang()


class _RecordingDb:
    """Wraps the real database, recording each system_tx and its executed statements."""

    def __init__(self, real):
        self._real = real
        self.transactions: list[list[str]] = []

    def system_tx(self):
        statements: list[str] = []
        self.transactions.append(statements)
        real_tx = self._real.system_tx()

        @asynccontextmanager
        async def wrapped():
            async with real_tx as conn:
                yield _RecordingConn(conn, statements)

        return wrapped()


class _RecordingConn:
    def __init__(self, conn, statements):
        self._conn = conn
        self._statements = statements

    async def execute(self, query, *args, **kwargs):
        self._statements.append(str(query))
        return await self._conn.execute(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


async def test_ready_hanging_database_gives_503_within_deadline(client, api_app, llm):
    llm(_llm_ok)
    api_app.state.db = _HangingDb()

    response = await asyncio.wait_for(client.get(READY), timeout=3.0)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["db"] == "error"
    assert body["migrations"] == "pending"
    assert body["worker"] == "stale"


async def test_ready_database_checks_share_one_bounded_transaction(client, api_app, db, llm):
    llm(_llm_ok)
    recording = _RecordingDb(db)
    api_app.state.db = recording

    response = await client.get(READY)

    assert response.status_code == 200
    assert len(recording.transactions) == 1
    assert "statement_timeout" in recording.transactions[0][0]


async def test_ready_skips_dependent_checks_when_database_fails(client, api_app, llm):
    llm(_llm_ok)
    calls = []

    class BrokenDb:
        def system_tx(self):
            calls.append(1)
            raise RuntimeError("down")

    api_app.state.db = BrokenDb()

    response = await client.get(READY)

    assert response.status_code == 503
    assert response.json()["migrations"] == "pending"
    assert response.json()["worker"] == "stale"
    assert len(calls) == 1


async def test_ready_slow_llm_is_unavailable_within_deadline(client, llm):
    async def drip(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json={"version": "0.0.0"})

    llm(drip)

    response = await asyncio.wait_for(client.get(READY), timeout=3.0)

    assert response.status_code == 200
    assert response.json()["llm"] == "unavailable"


async def test_ready_llm_server_error_is_unavailable(client, llm, caplog):
    llm(lambda request: httpx.Response(500))

    with caplog.at_level(logging.WARNING):
        response = await client.get(READY)

    assert response.status_code == 200
    assert response.json()["llm"] == "unavailable"


async def test_ready_llm_failure_logs_warning_without_traceback(client, llm, caplog):
    llm(_llm_down)

    with caplog.at_level(logging.WARNING):
        await client.get(READY)

    records = [r for r in caplog.records if getattr(r, "stage", None) == "llm"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == "readiness check failed"
    assert records[0].exc_info is None


async def test_metrics_still_renders_when_queue_count_fails(
    client, api_settings, monkeypatch, caplog
):
    api_settings.metrics_token = TOKEN
    from backend.api.routes import health

    async def broken(conn):
        raise RuntimeError("queue down")

    monkeypatch.setattr(health, "count_queued", broken)

    with caplog.at_level(logging.WARNING):
        response = await client.get(METRICS, headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert any(
        r.getMessage() == "metrics queue depth unavailable"
        and r.stage == "metrics_queue"
        and r.levelno == logging.WARNING
        for r in caplog.records
    )


async def test_metrics_hidden_without_token(client):
    response = await client.get(METRICS)

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_metrics_hidden_with_blank_token(client, api_settings):
    api_settings.metrics_token = "   "

    response = await client.get(METRICS, headers={"Authorization": "Bearer    "})

    assert response.status_code == 404


async def test_metrics_requires_token(client, api_settings):
    api_settings.metrics_token = TOKEN

    missing = await client.get(METRICS)
    wrong = await client.get(METRICS, headers={"Authorization": "Bearer nope"})
    bare = await client.get(METRICS, headers={"Authorization": TOKEN})

    for response in (missing, wrong, bare):
        assert response.status_code == 401
        assert response.json()["code"] == "unauthenticated"
        assert response.headers["www-authenticate"] == "Bearer"


async def test_metrics_reports_queue_depth(client, db, api_settings):
    api_settings.metrics_token = TOKEN
    async with db.system_tx() as conn:
        for _ in range(3):
            await queue.enqueue(conn, "ingest", {}, tenant_id=None)
    await client.get(LIVE)

    response = await client.get(METRICS, headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "jobs_queued 3.0" in response.text
    assert "http_requests_total" in response.text


async def test_applied_versions_empty_without_schema_migrations_table(fresh_db_url):
    async with await psycopg.AsyncConnection.connect(fresh_db_url, autocommit=True) as conn:
        assert await applied_versions(conn) == set()

    await migrate(fresh_db_url)

    async with await psycopg.AsyncConnection.connect(fresh_db_url, autocommit=True) as conn:
        assert await applied_versions(conn) == {m.version for m in discover()}
