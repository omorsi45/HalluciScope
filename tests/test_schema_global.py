import uuid

import psycopg
import pytest

from backend.db.pool import Database

pytestmark = pytest.mark.pg

GLOBAL_TABLES = {
    "tenants",
    "users",
    "memberships",
    "sessions",
    "api_keys",
    "auth_throttle",
    "jobs",
    "workers",
    "audit_log",
    "schema_migrations",
}


async def test_global_tables_exist(db):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select table_name from information_schema.tables where table_schema = 'public'"
        )
        names = {row["table_name"] for row in await cur.fetchall()}
    assert names >= GLOBAL_TABLES


async def test_email_is_case_insensitive_unique(db):
    insert = "insert into users (email, password_hash, display_name) values (%s, 'h', 'n')"
    async with db.system_tx() as conn:
        await conn.execute(insert, ("a@x.com",))
    with pytest.raises(psycopg.errors.UniqueViolation):
        async with db.system_tx() as conn:
            await conn.execute(insert, ("A@X.COM",))


async def test_job_status_check(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        async with db.system_tx() as conn:
            await conn.execute("insert into jobs (kind, status) values ('ingest', 'bogus')")


async def test_app_role_cannot_write_migrations(db):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with db.system_tx() as conn:
            await conn.execute(
                "insert into schema_migrations (version, name, checksum) values (999, 'x', 'y')"
            )


async def test_app_role_has_no_bypassrls(db):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select current_user as u, rolbypassrls as b from pg_roles "
            "where rolname = current_user"
        )
        row = await cur.fetchone()
    assert (row["u"], row["b"]) == ("eg_app", False)


async def test_tenant_setting_does_not_leak_across_pooled_connections(pg_urls):
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    setting = "select current_setting('app.tenant_id', true) as v"
    database = Database(pg_urls["app"], max_size=1)
    await database.open()
    try:
        async with database.tenant_tx(tenant_a) as conn:
            cur = await conn.execute(setting)
            assert (await cur.fetchone())["v"] == str(tenant_a)
        async with database.system_tx() as conn:
            cur = await conn.execute(setting)
            assert (await cur.fetchone())["v"] in ("", None)
        async with database.tenant_tx(tenant_b) as conn:
            cur = await conn.execute(setting)
            assert (await cur.fetchone())["v"] == str(tenant_b)
    finally:
        await database.close()
