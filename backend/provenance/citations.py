"""Check that the evidence and citations of an analysis still resolve to what is stored.

The schema already makes most of these problems impossible (composite foreign keys, immutable
versions and passages, the passage slice trigger). The check is the independent audit: it
re-reads the version text, so data damaged by a repair or a migration is reported instead of
shown as evidence. It reads only; the caller's connection comes from ``tenant_tx``.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from psycopg import AsyncConnection

ProblemKind = Literal[
    "span_outside_passage", "span_text_mismatch", "missing_passage", "citation_version_mismatch"
]

# A link names an exact (passage, version) pair, so the passage is joined on both: a pair that
# no passage row matches is `missing_passage`. A span is inside the passage when it lies in
# [start_char, end_char); the version text and the passage text must then agree on it.
# Offsets are 0-based, `substr` is 1-based, hence the `+ 1`.
_LINKS_SQL = (
    "select e.id as link_id, "
    "case "
    "when p.id is null then 'missing_passage' "
    "when e.span_start < p.start_char or e.span_end > p.end_char then 'span_outside_passage' "
    "when substr(v.text, e.span_start + 1, e.span_end - e.span_start) "
    "is distinct from "
    "substr(p.text, e.span_start - p.start_char + 1, e.span_end - e.span_start) "
    "then 'span_text_mismatch' "
    "end as problem "
    "from evidence_links e "
    "join claims c on c.id = e.claim_id "
    "left join passages p on p.id = e.passage_id and p.version_id = e.version_id "
    "left join document_versions v on v.id = e.version_id "
    "where c.analysis_id = %s "
    "order by c.ordinal, e.rank, e.id"
)

# A valid citation names a (passage, version) pair too. Here the passage is joined on its id
# alone, so a passage that belongs to another version is told apart from a missing one.
_CITATIONS_SQL = (
    "select c.id as citation_id, c.status, "
    "case "
    "when c.status <> 'valid' then null "
    "when p.id is null then 'missing_passage' "
    "when p.version_id <> c.version_id then 'citation_version_mismatch' "
    "end as problem "
    "from citations c "
    "left join passages p on p.id = c.passage_id "
    "where c.analysis_id = %s "
    "order by c.marker_start, c.number, c.id"
)


@dataclass(frozen=True)
class LinkProblem:
    """One broken evidence link, or one broken citation (``link_id`` is then the citation id)."""

    kind: ProblemKind
    link_id: UUID


@dataclass(frozen=True)
class CitationReport:
    n_links: int
    n_valid_links: int
    n_citations: int
    n_valid_citations: int  # `valid` citations without a problem; unknown markers are not valid
    n_unknown_markers: int
    problems: tuple[LinkProblem, ...]  # evidence links first, then citations


async def validate_analysis(conn: AsyncConnection, analysis_id: UUID) -> CitationReport:
    """Audit every evidence link and every citation of the analysis.

    A link is valid when its span lies inside its passage and the version text and the passage
    text agree on the span. A ``valid`` citation is valid when its passage belongs to its
    version. An ``unknown_marker`` citation has no passage to check and is only counted.
    """
    cur = await conn.execute(_LINKS_SQL, (analysis_id,))
    links = await cur.fetchall()
    cur = await conn.execute(_CITATIONS_SQL, (analysis_id,))
    citations = await cur.fetchall()

    problems = [LinkProblem(row["problem"], row["link_id"]) for row in links if row["problem"]]
    n_link_problems = len(problems)
    problems += [
        LinkProblem(row["problem"], row["citation_id"]) for row in citations if row["problem"]
    ]
    return CitationReport(
        n_links=len(links),
        n_valid_links=len(links) - n_link_problems,
        n_citations=len(citations),
        n_valid_citations=sum(
            1 for row in citations if row["status"] == "valid" and row["problem"] is None
        ),
        n_unknown_markers=sum(1 for row in citations if row["status"] == "unknown_marker"),
        problems=tuple(problems),
    )
