"""Resolve a passage id to the stored passage, its surroundings, and its standing today.

The text, offsets, and context always come from the passage's own (immutable) version, so a
citation shows exactly what the verdict was reached on. Whether that version is still the
current one, and where the same text lives in the current version, is reported next to it.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg import AsyncConnection

CONTEXT_CHARS = 300

# Offsets are 0-based Python indices; SQL `substr` is 1-based, hence the `+ 1`. Both count code
# points, so they agree for any text. The context before a passage is clipped at the version start.
_RESOLVE_SQL = (
    "select p.id as passage_id, p.version_id, v.version_number, v.effective_from, "
    "p.document_id, d.title as document_title, p.text, p.start_char, p.end_char, "
    "p.page_start, p.page_end, "
    "substr(v.text, p.start_char - least(p.start_char, %(context)s) + 1, "
    "least(p.start_char, %(context)s)) as context_before, "
    "substr(v.text, p.end_char + 1, %(context)s) as context_after, "
    "(v.id = d.current_version_id) is true as is_current_version, "
    "case when d.deleted_at is not null then null "
    "when v.id = d.current_version_id then p.id "
    "else (select q.id from passages q "
    "where q.version_id = d.current_version_id and q.text_sha256 = p.text_sha256 "
    "order by q.ordinal limit 1) end as current_equivalent_passage_id, "
    "case when v.id is distinct from d.current_version_id then cv.version_number end "
    "as superseded_by_version, "
    "p.collection_id, p.ordinal, p.section "
    "from passages p "
    "join document_versions v on v.id = p.version_id "
    "join documents d on d.id = p.document_id "
    "left join document_versions cv on cv.id = d.current_version_id "
    "where p.id = %(passage_id)s"
)


@dataclass(frozen=True)
class ResolvedPassage:
    passage_id: UUID
    version_id: UUID
    version_number: int
    effective_from: datetime
    document_id: UUID
    document_title: str
    text: str
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    context_before: str  # up to 300 characters of the version text ending at start_char
    context_after: str  # up to 300 characters of the version text starting at end_char
    is_current_version: bool
    current_equivalent_passage_id: UUID | None  # same text_sha256 in the current version
    superseded_by_version: int | None  # current version number when it is not this one
    collection_id: UUID
    ordinal: int
    section: str | None


async def resolve_passage(conn: AsyncConnection, passage_id: UUID) -> ResolvedPassage | None:
    """The passage with its context and current-version standing, or None when not visible.

    The connection comes from ``tenant_tx``, so another tenant's passage is simply absent.
    The caller decides whether the viewer may read the passage's collection.

    ``current_equivalent_passage_id`` is the passage of the document's current version with
    the same text hash (lowest ordinal), the passage itself when its version is current, and
    None when the document is deleted or the current version no longer holds that text.
    """
    cur = await conn.execute(_RESOLVE_SQL, {"passage_id": passage_id, "context": CONTEXT_CHARS})
    row = await cur.fetchone()
    return None if row is None else ResolvedPassage(**row)
