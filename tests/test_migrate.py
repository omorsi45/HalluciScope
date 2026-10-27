import asyncio
import uuid

import psycopg
import pytest

from backend.config import Settings
from backend.db.migrate import (
    MigrationChecksumError,
    MigrationError,
    discover,
    migrate,
    pending,
)
from tests.pg_fixtures import create_database, drop_database, role_url


def test_discover_orders_and_ignores_other_files(tmp_path):
    (tmp_path / "0002_b.sql").write_text("select 2;")
    (tmp_path / "0001_a.sql").write_text("select 1;")
    (tmp_path / "notes.txt").write_text("not a migration")

    found = discover(tmp_path)

    assert [m.version for m in found] == [1, 2]
    assert [m.name for m in found] == ["a", "b"]


def test_discover_rejects_duplicate_versions(tmp_path):
    (tmp_path / "0001_a.sql").write_text("select 1;")
    (tmp_path / "0001_b.sql").write_text("select 2;")

    with pytest.raises(MigrationError):
        discover(tmp_path)


async def test_migrate_refuses_an_empty_migrations_directory(tmp_path):
    # Unreachable on purpose: the error must come before any connection attempt.
    with pytest.raises(MigrationError, match="no migrations found"):
        await migrate("postgresql://eg_owner@localhost:1/none", tmp_path)


def test_settings_defaults():
    settings = Settings()
    assert settings.database_url == "postgresql://eg_app@localhost:54329/evidencegraph"
    assert (
        settings.migration_database_url == "postgresql://eg_owner@localhost:54329/evidencegraph"
    )
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.allow_registration is True
    assert settings.dev_insecure_cookies is False
    assert settings.metrics_token is None
    assert settings.worker_concurrency == 1
    assert settings.worker_metrics_port == 9101
    assert settings.log_level == "INFO"
    assert settings.session_ttl_days == 7


async def _table_exists(url: str, table: str) -> bool:
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        cur = await conn.execute("select to_regclass(%s)", (f"public.{table}",))
        row = await cur.fetchone()
        return row is not None and row[0] is not None


@pytest.mark.pg
async def test_migrate_applies_once(fresh_db_url, tmp_path):
    (tmp_path / "0001_one.sql").write_text("create table t1(id int);")
    (tmp_path / "0002_two.sql").write_text("create table t2(id int);")

    assert await pending(fresh_db_url, tmp_path) == [1, 2]
    assert await migrate(fresh_db_url, tmp_path) == [1, 2]
    assert await migrate(fresh_db_url, tmp_path) == []
    assert await pending(fresh_db_url, tmp_path) == []
    assert await _table_exists(fresh_db_url, "t1")
    assert await _table_exists(fresh_db_url, "t2")


@pytest.mark.pg
async def test_concurrent_migrate_on_fresh_database(fresh_db_url, tmp_path):
    (tmp_path / "0001_one.sql").write_text("create table t1(id int);")
    (tmp_path / "0002_two.sql").write_text("create table t2(id int);")

    extra = [f"eg_tmp_{uuid.uuid4().hex[:8]}" for _ in range(4)]
    urls = [fresh_db_url]
    try:
        for name in extra:
            await create_database(name)
            urls.append(role_url("eg_owner", name))

        for url in urls:
            results = await asyncio.gather(
                migrate(url, tmp_path), migrate(url, tmp_path), return_exceptions=True
            )
            assert not any(isinstance(r, BaseException) for r in results), results
            assert sorted(results) == [[], [1, 2]]
    finally:
        for name in extra:
            await drop_database(name)


@pytest.mark.pg
async def test_changed_checksum_refuses(fresh_db_url, tmp_path):
    first = tmp_path / "0001_one.sql"
    first.write_text("create table t1(id int);")
    await migrate(fresh_db_url, tmp_path)

    first.write_text("create table t1(id int, extra int);")

    with pytest.raises(MigrationChecksumError):
        await migrate(fresh_db_url, tmp_path)


@pytest.mark.pg
async def test_failed_migration_rolls_back(fresh_db_url, tmp_path):
    (tmp_path / "0001_one.sql").write_text("create table t1(id int);")
    (tmp_path / "0002_bad.sql").write_text("create table t2(id int); select from nope;")

    with pytest.raises(psycopg.Error):
        await migrate(fresh_db_url, tmp_path)

    async with await psycopg.AsyncConnection.connect(fresh_db_url, autocommit=True) as conn:
        cur = await conn.execute("select version from schema_migrations order by version")
        assert [r[0] for r in await cur.fetchall()] == [1]
    assert not await _table_exists(fresh_db_url, "t2")
