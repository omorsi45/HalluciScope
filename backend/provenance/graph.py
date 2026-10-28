"""The evidence graph of an analysis: its claims, cited answer sentences, and their sources.

Nodes are claims, answer sentences, passages, versions, and documents. Edges run from a claim
to a passage (``supports``, ``contradicts``, ``related``), from an answer sentence to a cited
passage (``cites``), from a passage to its version (``part_of``), from a version to its
document (``version_of``), and from a version to the document's current version
(``superseded_by``) when that is a different one.

A passage in a collection the viewer cannot read is a single node without label or details,
and its version and document are left out. Its source is fetched only for readable
collections, so withheld text never leaves the database. Reads only; the connection comes
from ``tenant_tx``.
"""

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from backend.db.repos import analyses

LABEL_CHARS = 120

_LINKS_SQL = (
    "select e.claim_id, e.passage_id, e.relation "
    "from evidence_links e join claims c on c.id = e.claim_id "
    "where c.analysis_id = %s"
)

# Only passages of readable collections are returned, so the others stay withheld.
_PASSAGES_SQL = (
    "select p.id, p.text, p.start_char, p.end_char, p.page_start, p.page_end, "
    "v.id as version_id, v.version_number, d.id as document_id, d.title, d.filename, "
    "d.current_version_id, cv.version_number as current_version_number "
    "from passages p "
    "join document_versions v on v.id = p.version_id "
    "join documents d on d.id = v.document_id "
    "left join document_versions cv on cv.id = d.current_version_id "
    "where p.id = any(%s::uuid[]) and p.collection_id = any(%s::uuid[])"
)

_COLLECTIONS_SQL = (
    "select p.collection_id from passages p where p.id in ("
    "select e.passage_id from evidence_links e join claims c on c.id = e.claim_id "
    "where c.analysis_id = %(analysis)s "
    "union select ci.passage_id from citations ci "
    "where ci.analysis_id = %(analysis)s and ci.passage_id is not null)"
)


def _cut(text: str) -> str:
    return text[:LABEL_CHARS]


async def passage_collection_ids(conn: AsyncConnection, analysis_id: UUID) -> set[UUID]:
    """The collections of every passage the analysis links to or cites."""
    cur = await conn.execute(_COLLECTIONS_SQL, {"analysis": analysis_id})
    return {row["collection_id"] for row in await cur.fetchall()}


async def build_graph(
    conn: AsyncConnection, analysis_id: UUID, readable: set[UUID]
) -> dict[str, list[dict[str, Any]]]:
    """The graph of the analysis as ``{"nodes": [...], "edges": [...]}``.

    ``readable`` holds the collections the viewer may read. Nodes are unique and sorted by id;
    edges are unique and sorted by (source, target, type), so equal inputs give equal output.
    Raises ``LookupError`` when the analysis is not visible.
    """
    analysis = await analyses.get_analysis(conn, analysis_id)
    if analysis is None:
        raise LookupError("analysis not found")
    answer = analysis["answer"] or ""
    claims = await analyses.get_claims(conn, analysis_id)
    cur = await conn.execute(_LINKS_SQL, (analysis_id,))
    links = await cur.fetchall()
    # An unknown marker names no passage, so it adds neither a sentence node nor an edge.
    citations = [
        citation
        for citation in await analyses.get_citations(conn, analysis_id)
        if citation["status"] == "valid"
    ]

    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, kind: str, label: str | None, data: dict[str, Any]) -> None:
        nodes.setdefault(node_id, {"id": node_id, "type": kind, "label": label, "data": data})

    for claim in claims:
        add_node(
            f"claim:{claim['id']}",
            "claim",
            _cut(claim["text"]),
            {"ordinal": claim["ordinal"], "verdict": claim["verdict"]},
        )
    for link in links:
        edges.add((f"claim:{link['claim_id']}", f"passage:{link['passage_id']}", link["relation"]))
    for citation in citations:
        start, end = citation["sentence_start"], citation["sentence_end"]
        sentence_id = f"sentence:{start}:{end}"
        add_node(sentence_id, "sentence", _cut(answer[start:end]), {"start": start, "end": end})
        edges.add((sentence_id, f"passage:{citation['passage_id']}", "cites"))

    passage_ids = {link["passage_id"] for link in links}
    passage_ids |= {citation["passage_id"] for citation in citations}
    cur = await conn.execute(_PASSAGES_SQL, (list(passage_ids), list(readable)))
    shown = {row["id"]: row for row in await cur.fetchall()}
    for passage_id in passage_ids:
        passage_node = f"passage:{passage_id}"
        row = shown.get(passage_id)
        if row is None:
            add_node(passage_node, "passage", None, {"redacted": True})
            continue
        add_node(
            passage_node,
            "passage",
            _cut(row["text"]),
            {
                "redacted": False,
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
            },
        )
        version_node = f"version:{row['version_id']}"
        document_node = f"document:{row['document_id']}"
        current_id = row["current_version_id"]
        add_node(
            version_node,
            "version",
            f"v{row['version_number']}",
            {
                "version_number": row["version_number"],
                "is_current": row["version_id"] == current_id,
            },
        )
        add_node(document_node, "document", row["title"], {"filename": row["filename"]})
        edges.add((passage_node, version_node, "part_of"))
        edges.add((version_node, document_node, "version_of"))
        if current_id is not None and current_id != row["version_id"]:
            current_node = f"version:{current_id}"
            add_node(
                current_node,
                "version",
                f"v{row['current_version_number']}",
                {"version_number": row["current_version_number"], "is_current": True},
            )
            edges.add((current_node, document_node, "version_of"))
            edges.add((version_node, current_node, "superseded_by"))

    return {
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "edges": [
            {"source": source, "target": target, "type": kind}
            for source, target, kind in sorted(edges)
        ],
    }
