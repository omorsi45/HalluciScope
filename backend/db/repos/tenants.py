"""Tenants and memberships repository."""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection


async def insert_tenant(conn: AsyncConnection, name: str) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into tenants (name) values (%s) returning id, name, created_at", (name,)
    )
    return await cur.fetchone()


async def add_membership(
    conn: AsyncConnection, tenant_id: UUID, user_id: UUID, role: str
) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into memberships (tenant_id, user_id, role) values (%s, %s, %s) "
        "returning tenant_id, user_id, role, created_at",
        (tenant_id, user_id, role),
    )
    return await cur.fetchone()


async def get_membership(
    conn: AsyncConnection, tenant_id: UUID, user_id: UUID
) -> dict[str, Any] | None:
    cur = await conn.execute(
        "select tenant_id, user_id, role, created_at from memberships "
        "where tenant_id = %s and user_id = %s",
        (tenant_id, user_id),
    )
    return await cur.fetchone()


async def list_memberships(conn: AsyncConnection, user_id: UUID) -> list[dict[str, Any]]:
    """Memberships of a user with the tenant name, oldest first."""
    cur = await conn.execute(
        "select m.tenant_id, m.user_id, m.role, m.created_at, t.name as tenant_name "
        "from memberships m join tenants t on t.id = m.tenant_id "
        "where m.user_id = %s order by m.created_at, m.tenant_id",
        (user_id,),
    )
    return await cur.fetchall()
