"""Value types shared by the ingestion pipeline.

All offsets are Python `str` indices (Unicode code points) into the canonical text.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageSpan:
    page: int
    start: int
    end: int


@dataclass(frozen=True)
class SectionSpan:
    start: int
    end: int
    path: str  # for example "Refunds > Exceptions"


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    pages: tuple[PageSpan, ...]
    sections: tuple[SectionSpan, ...]
    metadata: dict[str, str]
    parser: str
    parser_version: str


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int


@dataclass(frozen=True)
class PassageDraft:
    ordinal: int
    start: int
    end: int
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None
    text_sha256: bytes
