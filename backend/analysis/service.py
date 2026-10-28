"""Analysis service types."""

from dataclasses import dataclass
from uuid import UUID

from backend.generate.answer import CitationRef


@dataclass(frozen=True)
class CitationRecord(CitationRef):
    """A parsed citation plus the exact (passage, version) it resolved to.

    Both ids are None for an `unknown_marker`, and set together for a `valid` citation.
    """

    passage_id: UUID | None
    version_id: UUID | None
