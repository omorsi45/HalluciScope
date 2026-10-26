"""Audit log writer. Callers must never put secrets in `details`."""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from backend.obs.logging import request_id_var


async def record(
    conn: AsyncConnection,
    *,
    action: str,
    tenant_id: UUID | None,
    user_id: UUID | None,
    target_type: str | None = None,
    target_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        "insert into audit_log "
        "(tenant_id, user_id, action, target_type, target_id, request_id, details) "
        "values (%s, %s, %s, %s, %s, %s, %s)",
        (
            tenant_id,
            user_id,
            action,
            target_type,
            target_id,
            request_id_var.get(),
            Jsonb(details) if details is not None else None,
        ),
    )
