"""The visible passage set: access control plus bitemporal as-of (R9).

Live scopes read the ``is_current`` flag. Temporal scopes resolve one version per document with
``visible_version_ids`` and filter passages by those ids. Every retrieval query joins
``documents d on d.id = p.document_id`` and applies ``passage_filter``.
"""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from backend.retrieval.types import SearchScope

# known = transaction time, valid = valid time. now() is the start of the caller's transaction.
_VISIBLE_VERSIONS_SQL = (
    "with t as ("
    "select k.known, coalesce(%s::timestamptz, k.known) as valid "
    "from (select coalesce(%s::timestamptz, now()) as known) k) "
    "select distinct on (v.document_id) v.id "
    "from t, document_versions v join documents d on d.id = v.document_id "
    "where d.collection_id = any(%s::uuid[]) "
    "and d.created_at <= t.known "
    "and (d.deleted_at is null or d.deleted_at > t.known) "
    "and v.created_at <= t.known "
    "and v.effective_from <= t.valid "
    "order by v.document_id, v.version_number desc"
)


async def visible_version_ids(conn: AsyncConnection, scope: SearchScope) -> list[UUID] | None:
    """The version of each document that is valid at ``as_of`` as known at ``known_at``.

    ``known_at`` defaults to now and ``as_of`` to ``known_at``. Returns ``None`` for a live
    scope, whose visible passages are the ``is_current`` ones.
    """
    if scope.is_live:
        return None
    cur = await conn.execute(
        _VISIBLE_VERSIONS_SQL, (scope.as_of, scope.known_at, list(scope.collection_ids))
    )
    return [row["id"] for row in await cur.fetchall()]


def passage_filter(scope: SearchScope, version_ids: list[UUID] | None) -> tuple[str, list[Any]]:
    """A WHERE fragment over aliases ``p`` (passages) and ``d`` (documents), with its parameters."""
    if scope.is_live:
        return (
            "p.is_current and d.deleted_at is null and d.collection_id = any(%s::uuid[])",
            [list(scope.collection_ids)],
        )
    if version_ids is None:
        raise ValueError("a temporal scope needs the version ids from visible_version_ids")
    return "p.version_id = any(%s::uuid[])", [list(version_ids)]
