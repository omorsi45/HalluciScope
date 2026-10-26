"""Users repository. Functions take an open connection from `Database.system_tx()`."""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

_PUBLIC = "id, email, display_name, created_at, disabled_at"


async def insert_user(
    conn: AsyncConnection, email: str, password_hash: str, display_name: str
) -> dict[str, Any]:
    cur = await conn.execute(
        "insert into users (email, password_hash, display_name) values (%s, %s, %s) "
        f"returning {_PUBLIC}",
        (email, password_hash, display_name),
    )
    return await cur.fetchone()


async def get_user_by_email(conn: AsyncConnection, email: str) -> dict[str, Any] | None:
    cur = await conn.execute(f"select {_PUBLIC} from users where email = %s", (email,))
    return await cur.fetchone()


async def get_login_record(conn: AsyncConnection, email: str) -> dict[str, Any] | None:
    """The only lookup that returns password_hash; used by login."""
    cur = await conn.execute(
        f"select {_PUBLIC}, password_hash from users where email = %s", (email,)
    )
    return await cur.fetchone()


async def get_user(conn: AsyncConnection, user_id: UUID) -> dict[str, Any] | None:
    cur = await conn.execute(f"select {_PUBLIC} from users where id = %s", (user_id,))
    return await cur.fetchone()
