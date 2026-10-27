"""Postgres job queue: transactional enqueue, SKIP LOCKED claim, leases and retries.

Every function takes a connection and never opens its own transaction, so enqueue
joins the caller's transaction (a transactional outbox) and each claim or
completion commits with whatever the worker does around it.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

MAX_STRING_LENGTH = 100
MAX_LIST_ITEMS = 1000
_ERROR_CODE = re.compile(r"[a-z0-9_]{1,64}")


@dataclass(frozen=True)
class Job:
    id: int
    tenant_id: UUID | None
    kind: str
    payload: dict
    priority: int
    attempts: int
    max_attempts: int
    request_id: str | None


def _check_scalar(key: str, value: Any) -> None:
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"payload value for {key!r} must be finite")
        return
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ValueError(f"payload value for {key!r} exceeds {MAX_STRING_LENGTH} characters")
        return
    raise ValueError(f"payload value for {key!r} has an unsupported type")


def _validate_payload(payload: Mapping[str, Any]) -> None:
    """Ids and counts only (spec R4): scalars and flat lists of scalars, never text bodies."""
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("payload keys must be strings")
        if isinstance(value, list):
            if len(value) > MAX_LIST_ITEMS:
                raise ValueError(f"payload list for {key!r} exceeds {MAX_LIST_ITEMS} items")
            for item in value:
                _check_scalar(key, item)
        else:
            _check_scalar(key, value)


def _job_from_row(row: Mapping[str, Any]) -> Job:
    return Job(
        id=row["id"],
        tenant_id=row["tenant_id"],
        kind=row["kind"],
        payload=row["payload"],
        priority=row["priority"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        request_id=row["request_id"],
    )


async def enqueue(
    conn: AsyncConnection,
    kind: str,
    payload: Mapping[str, Any],
    *,
    tenant_id: UUID | None,
    priority: int = 0,
    idempotency_key: str | None = None,
    request_id: str | None = None,
    max_attempts: int = 3,
    run_after: datetime | None = None,
) -> int:
    _validate_payload(payload)
    cur = await conn.execute(
        "insert into jobs (tenant_id, kind, payload, priority, idempotency_key, request_id, "
        "max_attempts, run_after) "
        "values (%s, %s, %s, %s, %s, %s, %s, coalesce(%s::timestamptz, now())) "
        "on conflict (idempotency_key) do nothing returning id",
        (
            tenant_id,
            kind,
            Jsonb(dict(payload)),
            priority,
            idempotency_key,
            request_id,
            max_attempts,
            run_after,
        ),
    )
    row = await cur.fetchone()
    if row is None:
        # Only the idempotency_key conflict can suppress the insert, so the key is set.
        cur = await conn.execute(
            "select id from jobs where idempotency_key = %s", (idempotency_key,)
        )
        row = await cur.fetchone()
    return row["id"]


async def claim(conn: AsyncConnection, worker_id: str, kinds: Sequence[str]) -> Job | None:
    cur = await conn.execute(
        "update jobs set status = 'running', locked_by = %s, locked_at = now(), "
        "heartbeat_at = now(), attempts = attempts + 1, updated_at = now() "
        "where id = ("
        "select id from jobs where status = 'queued' and run_after <= now() "
        "and kind = any(%s::text[]) order by priority desc, id for update skip locked limit 1"
        ") returning *",
        (worker_id, list(kinds)),
    )
    row = await cur.fetchone()
    return None if row is None else _job_from_row(row)


async def heartbeat(conn: AsyncConnection, job_id: int, worker_id: str) -> bool:
    cur = await conn.execute(
        "update jobs set heartbeat_at = now(), updated_at = now() "
        "where id = %s and locked_by = %s and status = 'running' returning id",
        (job_id, worker_id),
    )
    return await cur.fetchone() is not None


async def complete(conn: AsyncConnection, job_id: int, worker_id: str) -> bool:
    cur = await conn.execute(
        "update jobs set status = 'succeeded', locked_by = null, locked_at = null, "
        "heartbeat_at = null, updated_at = now() "
        "where id = %s and locked_by = %s and status = 'running' returning id",
        (job_id, worker_id),
    )
    return await cur.fetchone() is not None


async def fail(
    conn: AsyncConnection,
    job_id: int,
    worker_id: str,
    error_code: str,
    *,
    permanent: bool = False,
) -> str | None:
    """Record a failure and return the new status, or None if the caller does not hold the lease."""
    if not _ERROR_CODE.fullmatch(error_code):
        raise ValueError("error_code must match [a-z0-9_]{1,64}")
    cur = await conn.execute(
        "update jobs set "
        "status = case when %(permanent)s then 'failed' "
        "when attempts < max_attempts then 'queued' else 'dead' end, "
        "run_after = case when not %(permanent)s and attempts < max_attempts "
        "then now() + make_interval(secs => 5 * power(4, attempts - 1)) else run_after end, "
        "last_error = %(error)s, locked_by = null, locked_at = null, heartbeat_at = null, "
        "updated_at = now() "
        "where id = %(id)s and locked_by = %(worker)s and status = 'running' returning status",
        {"permanent": permanent, "error": error_code, "id": job_id, "worker": worker_id},
    )
    row = await cur.fetchone()
    return None if row is None else row["status"]


async def release(conn: AsyncConnection, job_id: int, worker_id: str) -> bool:
    """Give a claimed job back without it counting as an attempt."""
    cur = await conn.execute(
        "update jobs set status = 'queued', attempts = attempts - 1, locked_by = null, "
        "locked_at = null, heartbeat_at = null, updated_at = now() "
        "where id = %s and locked_by = %s and status = 'running' returning id",
        (job_id, worker_id),
    )
    return await cur.fetchone() is not None


async def requeue_stale(conn: AsyncConnection, stale_after_s: float = 120.0) -> int:
    """Recover jobs whose worker stopped heartbeating; the lost attempt still counts."""
    cur = await conn.execute(
        "update jobs set "
        "status = case when attempts < max_attempts then 'queued' else 'dead' end, "
        "run_after = case when attempts < max_attempts then now() else run_after end, "
        "last_error = case when attempts < max_attempts then last_error "
        "else 'lease_expired' end, "
        "locked_by = null, locked_at = null, heartbeat_at = null, updated_at = now() "
        "where status = 'running' and heartbeat_at < now() - make_interval(secs => %s) "
        "returning id",
        (stale_after_s,),
    )
    return len(await cur.fetchall())


async def count_queued(conn: AsyncConnection) -> int:
    """Jobs waiting in status 'queued', including those delayed by run_after."""
    cur = await conn.execute("select count(*) as n from jobs where status = 'queued'")
    return (await cur.fetchone())["n"]
