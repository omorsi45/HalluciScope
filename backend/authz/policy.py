"""Collection authorization policy.

Every function takes a connection from ``tenant_tx(principal.tenant_id)``, so row level
security already confines the queries to the caller's tenant. Tenant owners and admins
have ``admin`` on every collection; everyone else needs an explicit grant. A collection
the caller cannot see is reported exactly like one that does not exist.
"""

from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection

from backend.api.errors import ApiError
from backend.auth.service import Principal

PERMISSION_RANK = {"read": 1, "write": 2, "admin": 3}

_TENANT_ADMIN_ROLES = frozenset({"owner", "admin"})
_COLLECTION_COLUMNS = "c.id, c.name, c.description, c.kind, c.created_by, c.created_at"


def is_tenant_admin(principal: Principal) -> bool:
    return principal.role in _TENANT_ADMIN_ROLES


def _not_found() -> ApiError:
    return ApiError(404, "collection_not_found", "Collection not found")


async def require_collection(
    conn: AsyncConnection,
    principal: Principal,
    collection_id: UUID,
    needed: Literal["read", "write", "admin"],
    *,
    kinds: Sequence[str] = ("normal",),
) -> dict[str, Any]:
    """Return the collection row plus the caller's effective ``permission``.

    A collection of a kind outside ``kinds`` is reported as not found, so by-id routes
    cannot reach quick or benchmark collections unless a caller asks for them.
    """
    cur = await conn.execute(
        f"select {_COLLECTION_COLUMNS}, g.permission as granted "
        "from collections c "
        "left join collection_grants g on g.collection_id = c.id and g.user_id = %s "
        "where c.id = %s and c.kind = any(%s)",
        (principal.user_id, collection_id, list(kinds)),
    )
    row = await cur.fetchone()
    if row is None:
        raise _not_found()
    granted = row.pop("granted")
    permission = "admin" if is_tenant_admin(principal) else granted
    if permission is None:
        raise _not_found()
    if PERMISSION_RANK[permission] < PERMISSION_RANK[needed]:
        raise ApiError(403, "insufficient_permission", "Insufficient permission on collection")
    row["permission"] = permission
    return row


async def readable_collection_ids(
    conn: AsyncConnection,
    principal: Principal,
    requested: Sequence[UUID] | None,
    *,
    kinds: Sequence[str] = ("normal",),
) -> tuple[UUID, ...]:
    """Ids the caller can read, restricted to ``kinds``.

    ``None`` returns every readable collection ordered by creation time. Otherwise every
    requested id must be readable or the call fails with 404, revealing nothing about
    which ids exist.
    """
    cur = await conn.execute(
        "select c.id from collections c "
        "left join collection_grants g on g.collection_id = c.id and g.user_id = %s "
        "where c.kind = any(%s) and (%s or g.user_id is not null) "
        "order by c.created_at, c.id",
        (principal.user_id, list(kinds), is_tenant_admin(principal)),
    )
    readable = [row["id"] for row in await cur.fetchall()]
    if requested is None:
        return tuple(readable)
    allowed = set(readable)
    ordered = tuple(dict.fromkeys(requested))
    if any(collection_id not in allowed for collection_id in ordered):
        raise _not_found()
    return ordered
