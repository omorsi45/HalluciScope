"""Dense search: cosine nearest neighbours over the visible passage set, on the HNSW index.

The visible filter is a WHERE clause on the same statement as the ordering. With
``hnsw.iterative_scan = relaxed_order`` pgvector keeps scanning the graph past ``ef_search``
until the filtered query has enough rows, so a selective filter does not starve the result.
Both settings are ``SET LOCAL``: they last until the end of the caller's transaction.
"""

from uuid import UUID

import numpy as np
from psycopg import AsyncConnection

from backend.retrieval.types import SearchScope
from backend.retrieval.visible import passage_filter

EF_SEARCH_MIN = 100
# pgvector rejects hnsw.ef_search above 1000; a larger limit is served by iterative scans.
EF_SEARCH_MAX = 1000

# The ORDER BY is the bare distance operator so that the planner can use the HNSW index.
_DENSE_SQL = """
select p.id as passage_id, (p.embedding <=> %s) as distance
from passages p
join documents d on d.id = p.document_id
where {visible}
order by p.embedding <=> %s
limit %s
"""


async def dense_search(
    conn: AsyncConnection,
    query_vec: np.ndarray,
    scope: SearchScope,
    *,
    limit: int,
    version_ids: list[UUID] | None,
    exact: bool = False,
) -> list[tuple[UUID, float]]:
    """Top ``limit`` passages by cosine similarity (``1 - distance``), best first, ties by id.

    ``version_ids`` is the result of ``visible_version_ids`` for the same scope (``None`` for a
    live scope). The caller owns the transaction: the ``SET LOCAL`` settings stay in effect for
    its later statements. ``exact`` disables index and bitmap scans, so the planner scans the
    filtered passages sequentially and the result is the true nearest set.
    """
    if not scope.is_live and not version_ids:
        return []
    if exact:
        await conn.execute("set local enable_indexscan = off")
        await conn.execute("set local enable_bitmapscan = off")
    else:
        ef_search = min(EF_SEARCH_MAX, max(EF_SEARCH_MIN, int(limit)))
        # An integer built here: SET does not take bind parameters.
        await conn.execute(f"set local hnsw.ef_search = {ef_search}")
        await conn.execute("set local hnsw.iterative_scan = relaxed_order")
    visible, visible_params = passage_filter(scope, version_ids)
    cur = await conn.execute(
        _DENSE_SQL.format(visible=visible),
        [query_vec, *visible_params, query_vec, limit],
    )
    hits = [(row["passage_id"], 1.0 - row["distance"]) for row in await cur.fetchall()]
    hits.sort(key=lambda hit: (-hit[1], hit[0]))
    return hits
