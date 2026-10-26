"""Users repository. Functions take an open connection from `Database.system_tx()`."""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection


async def insert_user(
    conn: AsyncConnection, email: str, password_hash: str, display_name: str
) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into users (email, password_hash, display_name) values (%s, %s, %s) "
        "returning id, email, display_name, created_at, disabled_at",
        (email, password_hash, display_name),
    )
    return await cur.fetchone()


async def get_user_by_email(conn: AsyncConnection, email: str) -> dict[str, Any] | None:
    cur = await conn.execute("select * from users where email = %s", (email,))
    return await cur.fetchone()


async def get_user(conn: AsyncConnection, user_id: UUID) -> dict[str, Any] | None:
    cur = await conn.execute("select * from users where id = %s", (user_id,))
    return await cur.fetchone()
