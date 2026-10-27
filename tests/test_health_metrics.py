import logging

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
