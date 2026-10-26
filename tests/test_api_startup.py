"""Lifespan resource handling in create_configured_app (no Postgres, no models)."""

from typing import ClassVar

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


class _FakeRepo:
    instances: ClassVar[list["_FakeRepo"]] = []

    def __init__(self, path: str) -> None:
        self.closed = False
        self.fail_close = False
        _FakeRepo.instances.append(self)

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("repo close failed")


@pytest.fixture
def patched(monkeypatch):
    _FakeDb.instances.clear()
    _FakeRepo.instances.clear()
    monkeypatch.setattr(startup, "Database", _FakeDb)
    monkeypatch.setattr(startup, "Repository", _FakeRepo)
    monkeypatch.setattr(startup, "configure_logging", lambda level: None)
    monkeypatch.setattr(startup, "get_embedding_model", lambda: object())
    monkeypatch.setattr(startup, "get_nli_model", lambda: (object(), object()))
    monkeypatch.setattr(startup, "Chunker", lambda **kw: object())
    monkeypatch.setattr(startup, "NLIVerifier", lambda **kw: object())
    monkeypatch.setattr(startup, "SimilarityVerifier", lambda **kw: object())
    monkeypatch.setattr(startup, "ConsistencyVerifier", lambda **kw: object())
    monkeypatch.setattr(startup, "EnsembleScorer", lambda **kw: object())
    monkeypatch.setattr(startup, "Pipeline", lambda **kw: object())


async def test_startup_failure_after_db_open_closes_pool(patched, monkeypatch):
    def boom():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(startup, "get_embedding_model", boom)
    app = startup.create_configured_app()
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    assert _FakeDb.instances[0].closed is True


async def test_shutdown_closes_every_resource_even_if_one_fails(patched, monkeypatch):
    app = startup.create_configured_app()
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            _FakeRepo.instances[0].fail_close = True
    assert _FakeRepo.instances[0].closed is True
    assert _FakeDb.instances[0].closed is True
    assert app.state.http_client.is_closed is True
