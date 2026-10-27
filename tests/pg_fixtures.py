"""Postgres helpers and fixtures for tests marked `pg`."""

import os
import uuid

import httpx
import psycopg
import pytest
import pytest_asyncio
from psycopg import sql
from psycopg.conninfo import make_conninfo

from backend.api.app import create_app
from backend.auth.service import AuthService
from backend.config import Settings
from backend.db.migrate import migrate
from backend.db.pool import Database

ADMIN_URL = os.environ.get(
    "HALLUCISCOPE_TEST_ADMIN_URL", "postgresql://postgres@localhost:54329/postgres"
)

__all__ = [
    "ADMIN_URL",
    "api_app",
    "api_settings",
    "client",
    "client2",
    "client3",
    "create_database",
    "db",
    "drop_database",
    "ensure_roles",
    "fresh_db_url",
    "pg_urls",
    "role_url",
]


def role_url(role: str, dbname: str) -> str:
    return make_conninfo(ADMIN_URL, user=role, dbname=dbname)


async def _connect(url: str) -> psycopg.AsyncConnection:
    try:
        return await psycopg.AsyncConnection.connect(url, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError:
        pytest.fail(
            f"Postgres not reachable at {ADMIN_URL}. Run scripts/dev-postgres.sh start",
            pytrace=False,
        )


async def ensure_roles() -> None:
    async with await _connect(ADMIN_URL) as conn:
        for role in ("eg_owner", "eg_app"):
            cur = await conn.execute("select 1 from pg_roles where rolname = %s", (role,))
            if await cur.fetchone() is None:
                await conn.execute(
                    sql.SQL("create role {} login nosuperuser nobypassrls").format(
                        sql.Identifier(role)
                    )
                )


async def create_database(name: str) -> None:
    async with await _connect(ADMIN_URL) as conn:
        await conn.execute(
            sql.SQL("create database {} owner {}").format(
                sql.Identifier(name), sql.Identifier("eg_owner")
            )
        )
    async with await _connect(role_url("postgres", name)) as conn:
        await conn.execute("create extension if not exists vector")
        await conn.execute("create extension if not exists citext")


async def drop_database(name: str) -> None:
    async with await _connect(ADMIN_URL) as conn:
        await conn.execute(
            sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name))
        )


@pytest.fixture
async def fresh_db_url():
    """Yield the eg_owner URL of a throwaway database, dropped afterwards."""
    await ensure_roles()
    name = f"eg_tmp_{uuid.uuid4().hex[:8]}"
    try:
        await create_database(name)
        yield role_url("eg_owner", name)
    finally:
        await drop_database(name)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_urls():
    """Migrated session database; yields the eg_owner and eg_app URLs."""
    await ensure_roles()
    name = f"eg_test_{uuid.uuid4().hex[:8]}"
    try:
        await create_database(name)
        owner = role_url("eg_owner", name)
        await migrate(owner)
        yield {"owner": owner, "app": role_url("eg_app", name)}
    finally:
        await drop_database(name)


@pytest_asyncio.fixture(loop_scope="session")
async def db(pg_urls):
    """App-role Database; every public table except schema_migrations is emptied after."""
    database = Database(pg_urls["app"], max_size=4)
    await database.open()
    try:
        yield database
    finally:
        await database.close()
        async with await _connect(pg_urls["owner"]) as conn:
            cur = await conn.execute(
                "select tablename from pg_tables "
                "where schemaname = 'public' and tablename <> 'schema_migrations'"
            )
            tables = [row[0] for row in await cur.fetchall()]
            if tables:
                await conn.execute(
                    sql.SQL("truncate {} restart identity cascade").format(
                        sql.SQL(", ").join(sql.Identifier(t) for t in tables)
                    )
                )


@pytest.fixture
def api_settings():
    return Settings(dev_insecure_cookies=True, cors_origins=["http://localhost:5173"])


@pytest.fixture
def api_app(db, api_settings):
    """App on the test database. ASGITransport skips lifespan, so state is set directly."""
    app = create_app(api_settings)
    app.state.db = db
    app.state.auth = AuthService(db, api_settings)
    return app


@pytest_asyncio.fixture(loop_scope="session")
async def client(api_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as http_client:
        yield http_client


@pytest_asyncio.fixture(loop_scope="session")
async def client2(api_app):
    """Second client with its own cookie jar, for two-user tests."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as http_client:
        yield http_client


@pytest_asyncio.fixture(loop_scope="session")
async def client3(api_app):
    """Third client with its own cookie jar, for three-user tests."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as http_client:
        yield http_client
