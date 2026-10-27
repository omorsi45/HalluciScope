"""Builds canonical text from parsed pieces while tracking page and section offsets."""

from backend.ingest.normalize import normalize
from backend.ingest.types import PageSpan, ParsedDocument, SectionSpan

_SEPARATOR = "\n\n"


class TextAssembler:
    """Joins normalized pieces with a blank line and records where pages and sections fall.

    Every returned or recorded offset indexes the final `ParsedDocument.text`.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self._pages: list[PageSpan] = []
        self._open_page: int | None = None
        self._sections: list[SectionSpan] = []
        self._section_path: str | None = None
        self._section_start: int | None = None

    def add(self, piece: str, *, page: int | None = None) -> tuple[int, int] | None:
        piece = normalize(piece)
        if not piece:
            return None
        if self._parts:
            self._parts.append(_SEPARATOR)
            self._length += len(_SEPARATOR)
        start = self._length
        end = start + len(piece)
        self._parts.append(piece)
        self._length = end

        if page is not None:
            if page == self._open_page:
                last = self._pages[-1]
                self._pages[-1] = PageSpan(last.page, last.start, end)
            else:
                self._pages.append(PageSpan(page, start, end))
        self._open_page = page

        if self._section_path is not None and self._section_start is None:
            self._section_start = start
        return start, end

    def start_section(self, path: str | None) -> None:
        self._close_section()
        self._section_path = path
        self._section_start = None

    def build(self, metadata: dict[str, str], parser: str, parser_version: str) -> ParsedDocument:
        self._close_section()
        return ParsedDocument(
            text="".join(self._parts),
            pages=tuple(self._pages),
            sections=tuple(self._sections),
            metadata=dict(metadata),
            parser=parser,
            parser_version=parser_version,
        )

    def _close_section(self) -> None:
        # A section that never received a piece has no start and is dropped.
        if self._section_path is not None and self._section_start is not None:
            self._sections.append(
                SectionSpan(self._section_start, self._length, self._section_path)
            )
        self._section_path = None
        self._section_start = None
