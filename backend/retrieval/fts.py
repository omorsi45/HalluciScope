"""The Postgres full-text baseline: an OR query over the visible set, ranked by cover density.

The query lexemes come from the same ``english`` analyzer as the stored ``tsv`` column and are
joined with ``|``, so a passage needs only one of them to match. ``ts_rank_cd`` then favors
passages where the matched terms are dense. A query without lexemes (only stopwords) has no
tsquery and matches nothing.
"""

from uuid import UUID

from psycopg import AsyncConnection

from backend.retrieval.types import SearchScope
from backend.retrieval.visible import passage_filter

# nullif keeps an empty lexeme list from becoming an empty tsquery (which raises a notice).
_FTS_SQL = """
with q as (
    select nullif(
        array_to_string(
            array(select quote_literal(lexeme) from unnest(to_tsvector('english', %s))),
            ' | '
        ),
        ''
    )::tsquery as query
)
select p.id as passage_id, ts_rank_cd(p.tsv, q.query)::float8 as score
from q, passages p
join documents d on d.id = p.document_id
where {visible} and p.tsv @@ q.query
order by score desc, p.id
limit %s
"""


async def fts_search(
    conn: AsyncConnection,
    query: str,
    scope: SearchScope,
    *,
    limit: int,
    version_ids: list[UUID] | None,
) -> list[tuple[UUID, float]]:
    """Top ``limit`` passages by ``ts_rank_cd``, best first, ties by passage id.

    ``version_ids`` is the result of ``visible_version_ids`` for the same scope (``None`` for a
    live scope).
    """
    if not scope.is_live and not version_ids:
        return []
    visible, visible_params = passage_filter(scope, version_ids)
    cur = await conn.execute(_FTS_SQL.format(visible=visible), [query, *visible_params, limit])
    return [(row["passage_id"], row["score"]) for row in await cur.fetchall()]
