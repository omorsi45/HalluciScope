"""Postgres helpers and fixtures for tests marked `pg`."""

import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

ADMIN_URL = os.environ.get(
    "HALLUCISCOPE_TEST_ADMIN_URL", "postgresql://postgres@localhost:54329/postgres"
)

__all__ = [
    "ADMIN_URL",
    "create_database",
    "drop_database",
    "ensure_roles",
    "fresh_db_url",
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
