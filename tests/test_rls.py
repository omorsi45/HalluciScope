import uuid

import psycopg
import pytest

pytestmark = pytest.mark.pg

# Tables without row-level security. Later plans extend this only with a spec reason.
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


async def _create_tenant(db, name: str) -> uuid.UUID:
    async with db.system_tx() as conn:
        cur = await conn.execute("insert into tenants (name) values (%s) returning id", (name,))
        return (await cur.fetchone())["id"]


async def _count_collections(conn) -> int:
    cur = await conn.execute("select count(*) as n from collections")
    return (await cur.fetchone())["n"]


async def test_collection_visible_only_in_own_tenant(db):
    tenant_a = await _create_tenant(db, "a")
    tenant_b = await _create_tenant(db, "b")
    async with db.tenant_tx(tenant_a) as conn:
        await conn.execute(
            "insert into collections (tenant_id, name) values (%s, %s)", (tenant_a, "docs")
        )
    async with db.tenant_tx(tenant_a) as conn:
        assert await _count_collections(conn) == 1
    async with db.tenant_tx(tenant_b) as conn:
        assert await _count_collections(conn) == 0


async def test_no_tenant_context_sees_no_rows(db):
    tenant_a = await _create_tenant(db, "a")
    async with db.tenant_tx(tenant_a) as conn:
        await conn.execute(
            "insert into collections (tenant_id, name) values (%s, %s)", (tenant_a, "docs")
        )
    async with db.system_tx() as conn:
        assert await _count_collections(conn) == 0


async def test_insert_for_other_tenant_rejected(db):
    tenant_a = await _create_tenant(db, "a")
    tenant_b = await _create_tenant(db, "b")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with db.tenant_tx(tenant_a) as conn:
            await conn.execute(
                "insert into collections (tenant_id, name) values (%s, %s)", (tenant_b, "docs")
            )


async def test_every_tenant_table_has_forced_rls(db):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select c.relname, c.relrowsecurity, c.relforcerowsecurity "
            "from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where c.relkind = 'r' and n.nspname = 'public'"
        )
        rows = await cur.fetchall()
    tenant_tables = {row["relname"]: row for row in rows if row["relname"] not in GLOBAL_TABLES}
    assert {"collections", "collection_grants"} <= tenant_tables.keys()
    unprotected = sorted(
        name
        for name, row in tenant_tables.items()
        if not (row["relrowsecurity"] and row["relforcerowsecurity"])
    )
    assert unprotected == []


async def _create_member(db, tenant_id: uuid.UUID, email: str) -> uuid.UUID:
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "insert into users (email, password_hash, display_name) values (%s, %s, %s) "
            "returning id",
            (email, "x", email),
        )
        user_id = (await cur.fetchone())["id"]
        await conn.execute(
            "insert into memberships (tenant_id, user_id, role) values (%s, %s, 'member')",
            (tenant_id, user_id),
        )
    return user_id


async def _create_collection(db, tenant_id: uuid.UUID, name: str) -> uuid.UUID:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into collections (tenant_id, name) values (%s, %s) returning id",
            (tenant_id, name),
        )
        return (await cur.fetchone())["id"]


_INSERT_GRANT = (
    "insert into collection_grants (tenant_id, collection_id, user_id, permission) "
    "values (%s, %s, %s, 'read')"
)


async def test_grant_for_other_tenants_collection_rejected(db):
    tenant_a = await _create_tenant(db, "a")
    tenant_b = await _create_tenant(db, "b")
    user_a = await _create_member(db, tenant_a, "a@example.com")
    collection_b = await _create_collection(db, tenant_b, "docs")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        async with db.tenant_tx(tenant_a) as conn:
            await conn.execute(_INSERT_GRANT, (tenant_a, collection_b, user_a))


async def test_grant_for_non_member_rejected(db):
    tenant_a = await _create_tenant(db, "a")
    tenant_b = await _create_tenant(db, "b")
    outsider = await _create_member(db, tenant_b, "outsider@example.com")
    collection_a = await _create_collection(db, tenant_a, "docs")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        async with db.tenant_tx(tenant_a) as conn:
            await conn.execute(_INSERT_GRANT, (tenant_a, collection_a, outsider))


async def test_removing_membership_removes_grants(db):
    tenant_a = await _create_tenant(db, "a")
    user_a = await _create_member(db, tenant_a, "a@example.com")
    collection_a = await _create_collection(db, tenant_a, "docs")
    async with db.tenant_tx(tenant_a) as conn:
        await conn.execute(_INSERT_GRANT, (tenant_a, collection_a, user_a))
    async with db.system_tx() as conn:
        await conn.execute(
            "delete from memberships where tenant_id = %s and user_id = %s", (tenant_a, user_a)
        )
    async with db.tenant_tx(tenant_a) as conn:
        cur = await conn.execute("select count(*) as n from collection_grants")
        assert (await cur.fetchone())["n"] == 0
