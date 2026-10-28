"""Lifespan resource handling in create_configured_app (no Postgres, no models)."""

from typing import ClassVar

import httpx
import pytest

from backend.api import startup


class _FakeDb:
    instances: ClassVar[list["_FakeDb"]] = []

    def __init__(self, url: str) -> None:
        self.closed = False
        _FakeDb.instances.append(self)

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class _FailingCloseClient(httpx.AsyncClient):
    async def aclose(self) -> None:
        await super().aclose()
        raise RuntimeError("http close failed")


@pytest.fixture
def patched(monkeypatch):
    _FakeDb.instances.clear()
    monkeypatch.setattr(startup, "Database", _FakeDb)
    monkeypatch.setattr(startup, "configure_logging", lambda level: None)
    monkeypatch.setattr(startup, "load_embedder", lambda settings: object())


async def test_startup_failure_after_db_open_closes_pool(patched, monkeypatch):
    def boom(db, settings):
        raise RuntimeError("auth service failed")

    monkeypatch.setattr(startup, "AuthService", boom)
    app = startup.create_configured_app()
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    assert _FakeDb.instances[0].closed is True


async def test_startup_loads_the_search_embedder(patched, monkeypatch):
    embedder = object()
    loads = []

    def load(settings):
        loads.append(settings)
        return embedder

    monkeypatch.setattr(startup, "load_embedder", load)
    app = startup.create_configured_app()
    async with app.router.lifespan_context(app):
        assert app.state.embedder is embedder
    assert loads == [app.state.settings]


async def test_embedder_load_failure_closes_pool(patched, monkeypatch):
    def boom(settings):
        raise OSError("model files missing")

    monkeypatch.setattr(startup, "load_embedder", boom)
    app = startup.create_configured_app()
    with pytest.raises(OSError):
        async with app.router.lifespan_context(app):
            pass
    assert _FakeDb.instances[0].closed is True


async def test_startup_opens_no_sqlite_database(patched, monkeypatch, tmp_path):
    legacy = tmp_path / "legacy.db"
    monkeypatch.setenv("HALLUCISCOPE_DB_PATH", str(legacy))
    app = startup.create_configured_app()
    async with app.router.lifespan_context(app):
        pass
    assert not legacy.exists()


async def test_shutdown_closes_the_http_client_and_the_database(patched):
    app = startup.create_configured_app()
    async with app.router.lifespan_context(app):
        assert app.state.http is app.state.http_client
        assert app.state.http_client.is_closed is False
    assert app.state.http_client.is_closed is True
    assert _FakeDb.instances[0].closed is True


async def test_shutdown_closes_the_database_even_if_the_http_client_fails_to_close(
    patched, monkeypatch
):
    monkeypatch.setattr(startup.httpx, "AsyncClient", _FailingCloseClient)
    app = startup.create_configured_app()
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    assert app.state.http_client.is_closed is True
    assert _FakeDb.instances[0].closed is True
