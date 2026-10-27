"""API keys repository. Only the sha256 of a key is stored."""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

_PUBLIC = "id, tenant_id, user_id, name, prefix, created_at, last_used_at, revoked_at"


async def insert_api_key(
    conn: AsyncConnection,
    tenant_id: UUID,
    user_id: UUID,
    name: str,
    prefix: str,
    key_hash: bytes,
) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into api_keys (tenant_id, user_id, name, prefix, key_hash) "
        f"values (%s, %s, %s, %s, %s) returning {_PUBLIC}",
        (tenant_id, user_id, name, prefix, key_hash),
    )
    return await cur.fetchone()


async def get_active_by_hash(conn: AsyncConnection, key_hash: bytes) -> dict[str, Any] | None:
    """The key if not revoked and its user is not disabled, with the user's identity."""
    cur = await conn.execute(
        "select k.id, k.tenant_id, k.user_id, k.name, k.prefix, u.email, u.display_name "
        "from api_keys k join users u on u.id = k.user_id "
        "where k.key_hash = %s and k.revoked_at is null and u.disabled_at is null",
        (key_hash,),
    )
    return await cur.fetchone()


async def get(conn: AsyncConnection, key_id: UUID, tenant_id: UUID) -> dict[str, Any] | None:
    cur = await conn.execute(
        f"select {_PUBLIC} from api_keys where id = %s and tenant_id = %s",
        (key_id, tenant_id),
    )
    return await cur.fetchone()


async def list_for_user(
    conn: AsyncConnection, tenant_id: UUID, user_id: UUID
) -> list[dict[str, Any]]:
    """Active (not revoked) keys only."""
    cur = await conn.execute(
        f"select {_PUBLIC} from api_keys where tenant_id = %s and user_id = %s "
        "and revoked_at is null order by created_at, id",
        (tenant_id, user_id),
    )
    return await cur.fetchall()


async def revoke(conn: AsyncConnection, key_id: UUID, tenant_id: UUID) -> bool:
    cur = await conn.execute(
        "update api_keys set revoked_at = now() "
        "where id = %s and tenant_id = %s and revoked_at is null",
        (key_id, tenant_id),
    )
    return cur.rowcount > 0


async def touch(conn: AsyncConnection, key_id: UUID) -> None:
    """Record use, at most once a minute, so busy keys do not rewrite their row per request."""
    await conn.execute(
        "update api_keys set last_used_at = now() where id = %s "
        "and (last_used_at is null or last_used_at < now() - interval '60 seconds')",
        (key_id,),
    )
