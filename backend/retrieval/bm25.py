"""Original BM25: a Python reference and one SQL statement over the visible passage set.

``idf = ln(1 + (N - df + 0.5) / (df + 0.5))`` and a term scores
``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``. N, df and avgdl are computed over
the caller's visible passages only, so passages in collections the caller cannot read never
change a score. Everything is float8, which keeps SQL and Python equal to far better than 1e-9.
"""

import math
from collections.abc import Mapping, Sequence
from uuid import UUID

from psycopg import AsyncConnection

from backend.retrieval.types import SearchScope
from backend.retrieval.visible import passage_filter


def bm25_term_score(
    tf: int,
    df: int,
    n_docs: int,
    doc_len: int,
    avg_doc_len: float,
    k1: float = 0.9,
    b: float = 0.4,
) -> float:
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))


def bm25_reference(
    query_terms: Sequence[str],
    term_freqs: Mapping[str, Mapping[str, int]],
    doc_lens: Mapping[str, int],
    k1: float = 0.9,
    b: float = 0.4,
) -> dict[str, float]:
    """Scores of the passages containing at least one query term.

    ``doc_lens`` lists every passage of the set, so N and avgdl are taken over all of them;
    ``term_freqs[passage]`` maps term to tf and may omit passages without terms. A repeated
    query term counts once.
    """
    n_docs = len(doc_lens)
    if n_docs == 0:
        return {}
    avg_doc_len = sum(doc_lens.values()) / n_docs
    terms = sorted(set(query_terms))
    df = {term: sum(1 for freqs in term_freqs.values() if freqs.get(term, 0) > 0) for term in terms}
    scores: dict[str, float] = {}
    for passage, freqs in term_freqs.items():
        score = 0.0
        matched = False
        for term in terms:
            tf = freqs.get(term, 0)
            if tf > 0:
                matched = True
                score += bm25_term_score(
                    tf, df[term], n_docs, doc_lens[passage], avg_doc_len, k1, b
                )
        if matched:
            scores[passage] = score
    return scores


async def query_terms(conn: AsyncConnection, query: str) -> list[str]:
    """Distinct lexemes of the query, sorted: the analyzer that built the index."""
    cur = await conn.execute(
        "select lexeme from unnest(to_tsvector('english', %s)) order by lexeme", (query,)
    )
    return [row["lexeme"] for row in await cur.fetchall()]


# One statement: the visible set, its statistics, per-term df, then the scored sum.
_BM25_SQL = """
with vis as (
    select p.id, p.token_count
    from passages p
    join documents d on d.id = p.document_id
    where {visible}
),
stats as (
    select count(*)::float8 as n, avg(token_count)::float8 as avgdl from vis
),
df as (
    select t.term, count(*)::float8 as df
    from passage_terms t
    join vis on vis.id = t.passage_id
    where t.term = any(%s::text[])
    group by t.term
)
select t.passage_id,
    sum(
        ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5))
        * t.tf::float8 * (%s::float8 + 1)
        / (t.tf::float8 + %s::float8
            * (1 - %s::float8 + %s::float8 * vis.token_count::float8 / stats.avgdl))
    )::float8 as score
from passage_terms t
join vis on vis.id = t.passage_id
join df on df.term = t.term
cross join stats
where t.term = any(%s::text[])
group by t.passage_id
order by score desc, t.passage_id
limit %s
"""


async def bm25_search(
    conn: AsyncConnection,
    query: str,
    scope: SearchScope,
    *,
    limit: int,
    version_ids: list[UUID] | None,
    k1: float = 0.9,
    b: float = 0.4,
) -> list[tuple[UUID, float]]:
    """Top ``limit`` passages by BM25, best first, ties by passage id.

    ``version_ids`` is the result of ``visible_version_ids`` for the same scope (``None`` for a
    live scope).
    """
    terms = await query_terms(conn, query)
    if not terms or (not scope.is_live and not version_ids):
        return []
    visible, visible_params = passage_filter(scope, version_ids)
    cur = await conn.execute(
        _BM25_SQL.format(visible=visible),
        [*visible_params, terms, k1, k1, b, b, terms, limit],
    )
    return [(row["passage_id"], row["score"]) for row in await cur.fetchall()]
