"""Tenant member and audit listing queries. Connections come from ``Database.system_tx()``."""

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection


async def list_members(conn: AsyncConnection, tenant_id: UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "select u.id as user_id, u.email, u.display_name, m.role, m.created_at "
        "from memberships m join users u on u.id = m.user_id "
        "where m.tenant_id = %s order by m.created_at, m.user_id",
        (tenant_id,),
    )
    return await cur.fetchall()


async def member_out(conn: AsyncConnection, tenant_id: UUID, user_id: UUID) -> dict[str, Any]:
    cur = await conn.execute(
        "select u.id as user_id, u.email, u.display_name, m.role, m.created_at "
        "from memberships m join users u on u.id = m.user_id "
        "where m.tenant_id = %s and m.user_id = %s",
        (tenant_id, user_id),
    )
    return await cur.fetchone()


async def lock_owner_ids(conn: AsyncConnection, tenant_id: UUID) -> list[UUID]:
    """Lock the tenant's owner memberships so concurrent removals serialize."""
    cur = await conn.execute(
        "select user_id from memberships where tenant_id = %s and role = 'owner' for update",
        (tenant_id,),
    )
    return [row["user_id"] for row in await cur.fetchall()]


async def delete_membership(conn: AsyncConnection, tenant_id: UUID, user_id: UUID) -> None:
    """Grants of the user in the tenant go with it (on delete cascade)."""
    await conn.execute(
        "delete from memberships where tenant_id = %s and user_id = %s", (tenant_id, user_id)
    )


async def list_audit(
    conn: AsyncConnection,
    tenant_id: UUID,
    *,
    limit: int,
    after: tuple[datetime, int] | None,
) -> list[dict[str, Any]]:
    """Audit rows newest first. Returns up to ``limit + 1`` rows to detect another page."""
    cursor_sql = "and (at, id) < (%s, %s::bigint) " if after else ""
    params: list[Any] = [tenant_id]
    if after:
        params.extend(after)
    params.append(limit + 1)
    cur = await conn.execute(
        "select id, action, user_id, target_type, target_id, at, details "
        "from audit_log where tenant_id = %s "
        f"{cursor_sql}"
        "order by at desc, id desc limit %s",
        params,
    )
    return await cur.fetchall()
