"""Collections and grants repository. Connections come from ``tenant_tx``."""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

_COLUMNS = "c.id, c.name, c.description, c.kind, c.created_at"


async def insert_collection(
    conn: AsyncConnection, tenant_id: UUID, name: str, description: str, created_by: UUID
) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into collections (tenant_id, name, description, created_by) "
        "values (%s, %s, %s, %s) returning id, name, description, kind, created_at",
        (tenant_id, name, description, created_by),
    )
    return await cur.fetchone()


async def update_collection(
    conn: AsyncConnection, collection_id: UUID, name: str | None, description: str | None
) -> dict[str, Any]:
    cur = await conn.execute(
        "update collections set name = coalesce(%s, name), "
        "description = coalesce(%s, description) where id = %s "
        "returning id, name, description, kind, created_at",
        (name, description, collection_id),
    )
    return await cur.fetchone()


async def delete_collection(conn: AsyncConnection, collection_id: UUID) -> None:
    await conn.execute("delete from collections where id = %s", (collection_id,))


async def list_collections(
    conn: AsyncConnection,
    user_id: UUID,
    *,
    is_tenant_admin: bool,
    limit: int,
    after: tuple[Any, UUID] | None,
) -> list[dict[str, Any]]:
    """Readable ``normal`` collections, newest first, with the effective permission.

    Returns up to ``limit + 1`` rows so the caller can tell whether another page exists.
    """
    cursor_sql = "and (c.created_at, c.id) < (%s, %s::uuid) " if after else ""
    params: list[Any] = [is_tenant_admin, user_id, is_tenant_admin]
    if after:
        params.extend(after)
    params.append(limit + 1)
    cur = await conn.execute(
        f"select {_COLUMNS}, "
        "case when %s then 'admin' else g.permission end as permission "
        "from collections c "
        "left join collection_grants g on g.collection_id = c.id and g.user_id = %s "
        "where c.kind = 'normal' and (%s or g.user_id is not null) "
        f"{cursor_sql}"
        "order by c.created_at desc, c.id desc limit %s",
        params,
    )
    return await cur.fetchall()


async def insert_grant(
    conn: AsyncConnection, tenant_id: UUID, collection_id: UUID, user_id: UUID, permission: str
) -> None:
    await conn.execute(
        "insert into collection_grants (tenant_id, collection_id, user_id, permission) "
        "values (%s, %s, %s, %s)",
        (tenant_id, collection_id, user_id, permission),
    )


async def list_grants(conn: AsyncConnection, collection_id: UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "select g.user_id, u.email, u.display_name, g.permission "
        "from collection_grants g join users u on u.id = g.user_id "
        "where g.collection_id = %s order by g.created_at, g.user_id",
        (collection_id,),
    )
    return await cur.fetchall()


async def replace_grants(
    conn: AsyncConnection,
    tenant_id: UUID,
    collection_id: UUID,
    grants: list[tuple[UUID, str]],
) -> None:
    await conn.execute("delete from collection_grants where collection_id = %s", (collection_id,))
    for user_id, permission in grants:
        await insert_grant(conn, tenant_id, collection_id, user_id, permission)


async def count_members(conn: AsyncConnection, tenant_id: UUID, user_ids: list[UUID]) -> int:
    """How many of ``user_ids`` are members of the tenant (memberships is global)."""
    cur = await conn.execute(
        "select count(*) as n from memberships where tenant_id = %s and user_id = any(%s)",
        (tenant_id, user_ids),
    )
    return (await cur.fetchone())["n"]
