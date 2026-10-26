"""Sessions repository. Time is always passed in, never read from the database clock."""

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection


async def insert_session(
    conn: AsyncConnection,
    token_hash: bytes,
    csrf_hash: bytes,
    user_id: UUID,
    tenant_id: UUID,
    expires_at: datetime,
) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into sessions (token_hash, csrf_hash, user_id, tenant_id, expires_at) "
        "values (%s, %s, %s, %s, %s) "
        "returning id, user_id, tenant_id, created_at, expires_at",
        (token_hash, csrf_hash, user_id, tenant_id, expires_at),
    )
    return await cur.fetchone()


async def get_active_session(
    conn: AsyncConnection, token_hash: bytes, now: datetime
) -> dict[str, Any] | None:
    """The session if not revoked, not expired, and its user is not disabled."""
    cur = await conn.execute(
        "select s.id, s.csrf_hash, s.user_id, s.tenant_id, s.expires_at, "
        "u.email, u.display_name "
        "from sessions s join users u on u.id = s.user_id "
        "where s.token_hash = %s and s.revoked_at is null and s.expires_at > %s "
        "and u.disabled_at is null",
        (token_hash, now),
    )
    return await cur.fetchone()


async def revoke_session(conn: AsyncConnection, token_hash: bytes) -> None:
    await conn.execute(
        "update sessions set revoked_at = now() where token_hash = %s and revoked_at is null",
        (token_hash,),
    )


async def set_session_tenant(conn: AsyncConnection, session_id: UUID, tenant_id: UUID) -> None:
    await conn.execute("update sessions set tenant_id = %s where id = %s", (tenant_id, session_id))
