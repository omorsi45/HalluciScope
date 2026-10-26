"""Login failure counters, one row per (key, window)."""

from datetime import datetime

from psycopg import AsyncConnection


async def get_failures(conn: AsyncConnection, key: str, window_start: datetime) -> int:
    cur = await conn.execute(
        "select failures from auth_throttle where key = %s and window_start = %s",
        (key, window_start),
    )
    row = await cur.fetchone()
    return row["failures"] if row else 0


async def record_failure(conn: AsyncConnection, key: str, window_start: datetime) -> None:
    await conn.execute(
        "insert into auth_throttle (key, window_start, failures) values (%s, %s, 1) "
        "on conflict (key, window_start) do update set failures = auth_throttle.failures + 1",
        (key, window_start),
    )


async def clear(conn: AsyncConnection, key: str, window_start: datetime) -> None:
    await conn.execute(
        "delete from auth_throttle where key = %s and window_start = %s", (key, window_start)
    )
