"""Login attempt counters, one row per (key, window).

Attempts are reserved with a single conditional upsert, so concurrent logins cannot
slip past a limit between a read and a later write.
"""

from datetime import datetime

from psycopg import AsyncConnection


async def reserve(conn: AsyncConnection, key: str, window_start: datetime, limit: int) -> bool:
    """Count one attempt unless the key already has `limit` of them. True if reserved."""
    cur = await conn.execute(
        "insert into auth_throttle (key, window_start, failures) values (%s, %s, 1) "
        "on conflict (key, window_start) do update set failures = auth_throttle.failures + 1 "
        "where auth_throttle.failures < %s returning failures",
        (key, window_start, limit),
    )
    return await cur.fetchone() is not None


async def release(conn: AsyncConnection, key: str, window_start: datetime) -> None:
    """Give back one reserved attempt."""
    await conn.execute(
        "update auth_throttle set failures = failures - 1 "
        "where key = %s and window_start = %s and failures > 0",
        (key, window_start),
    )


async def clear(conn: AsyncConnection, key: str, window_start: datetime) -> None:
    await conn.execute(
        "delete from auth_throttle where key = %s and window_start = %s", (key, window_start)
    )
