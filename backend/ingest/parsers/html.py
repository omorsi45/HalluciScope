"""HTML parser: extracts readable text, headings as sections, lists, and tables.

Built on the standard library `HTMLParser`, a streaming tokenizer, so there is no tree and no
recursion: nesting depth is irrelevant to the cost. Every piece goes through `TextAssembler`,
which owns normalization and exact offsets.

Uploads are untrusted and neither a regex nor this loop can be interrupted by the parse
timeout, so all per-tag work is constant (or amortized constant for the skip stack).
"""

import re
from html.parser import HTMLParser

from backend.ingest.assemble import TextAssembler
from backend.ingest.normalize import normalize
from backend.ingest.parsers.text import decode_utf8
from backend.ingest.types import ParsedDocument

# Elements whose content is not page text. `<title>` inside `<head>` is read separately.
_SKIPPED = frozenset({"script", "style", "noscript", "template", "svg", "head"})
# The only elements allowed inside `<head>`. Anything else implies an omitted `</head>`.
_HEAD_CONTENT = frozenset(
    {"title", "meta", "link", "base", "script", "style", "noscript", "template", "head"}
)
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_CELLS = frozenset({"td", "th"})
_ROW_BOUNDARIES = frozenset({"tr", "table"})
_BLOCKS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "nav",
        "aside",
        "ul",
        "ol",
        "li",
        "table",
        "tr",
        "pre",
        "blockquote",
        "dl",
        "dt",
        "dd",
        "figure",
        "figcaption",
        "hr",
        "form",
    }
)

# HTML collapses runs of these (and only these) to one space outside `<pre>`.
_HTML_SPACE = re.compile(r"[ \t\r\n\f]+")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.assembler = TextAssembler()
        self.metadata: dict[str, str] = {}
        self._buf: list[str] = []
        self._skip_stack: list[str] = []
        self._skip_counts: dict[str, int] = {}
        self._pre = 0
        self._in_title = False
        self._title: list[str] = []
        self._heading: int | None = None
        self._headings: list[tuple[int, str]] = []
        self._row: list[str] | None = None
        self._item_prefix = False

    # -- tokenizer callbacks -------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._in_title:
            return
        if (
            len(self._skip_stack) == 1
            and self._skip_stack[0] == "head"
            and tag not in _HEAD_CONTENT
        ):
            # `</head>` may be omitted; the first body element ends the head.
            self._skip_stack.clear()
            self._skip_counts.clear()
        if tag == "title":
            # A title nested in svg, template, or noscript is not the document title.
            if len(self._skip_stack) == self._skip_counts.get("head", 0):
                self._in_title = True
            return
        if tag in _SKIPPED:
            self._skip_stack.append(tag)
            self._skip_counts[tag] = self._skip_counts.get(tag, 0) + 1
            return
        if self._skip_stack:
            return

        if tag == "br":
            self._buf.append("\n")
        elif tag in _HEADINGS:
            self._flush()
            self._end_row()
            self._heading = int(tag[1])
        elif tag in _CELLS:
            self._flush()
            if self._row is None:
                self._row = []
        elif tag in _BLOCKS:
            self._flush()
            if tag in _ROW_BOUNDARIES:
                self._end_row()
            elif tag == "pre":
                self._pre += 1
            elif tag == "li":
                self._item_prefix = True

    def handle_endtag(self, tag: str) -> None:
        if self._in_title:
            if tag == "title":
                self._end_title()
            return
        if tag in _SKIPPED:
            self._end_skipped(tag)
            return
        if self._skip_stack:
            return

        if tag in _HEADINGS or tag in _CELLS:
            self._flush()
        elif tag in _BLOCKS:
            self._flush()
            if tag in _ROW_BOUNDARIES:
                self._end_row()
            elif tag == "pre":
                self._pre = max(0, self._pre - 1)
            elif tag in ("li", "ul", "ol"):
                self._item_prefix = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif not self._skip_stack:
            self._buf.append(data if self._pre else _HTML_SPACE.sub(" ", data))

    # -- state transitions ---------------------------------------------------

    def finish(self) -> ParsedDocument:
        self._flush()
        self._end_row()
        return self.assembler.build(self.metadata, "html", "html/1")

    def _end_title(self) -> None:
        self._in_title = False
        title = normalize(" ".join("".join(self._title).split()))
        self._title.clear()
        if title and "title" not in self.metadata:
            self.metadata["title"] = title

    def _end_skipped(self, tag: str) -> None:
        """Close the innermost open `tag` and everything opened inside it.

        Each stack entry is pushed and popped once, so a run of mismatched end tags stays
        linear. An end tag with no open counterpart is ignored.
        """
        if self._skip_counts.get(tag, 0) == 0:
            return
        while True:
            top = self._skip_stack.pop()
            self._skip_counts[top] -= 1
            if top == tag:
                return

    def _take(self) -> str:
        """Return the buffered inline text as canonical text and empty the buffer."""
        text = normalize("".join(self._buf))
        self._buf.clear()
        if text and not self._pre:
            # `<br>` leaves a space on the next line; `normalize` only removes trailing ones.
            text = "\n".join(line.lstrip(" ") for line in text.split("\n"))
        if text and self._item_prefix:
            self._item_prefix = False
            text = f"- {text}"
        return text

    def _flush(self) -> None:
        """End the current block: a heading, a table cell, or an ordinary paragraph."""
        if self._heading is not None:
            self._end_heading()
        elif self._row is not None:
            self._end_cell()
        else:
            text = self._take()
            if text:
                self.assembler.add(text)

    def _end_cell(self) -> None:
        text = self._take()
        if text and self._row is not None:
            self._row.append(text)

    def _end_row(self) -> None:
        if self._row is None:
            return
        self._end_cell()
        row, self._row = self._row, None
        if row:
            self.assembler.add("; ".join(row))

    def _end_heading(self) -> None:
        level = self._heading
        self._heading = None
        # A heading is one line: line breaks and runs of whitespace become single spaces.
        text = normalize(" ".join("".join(self._buf).split()))
        self._buf.clear()
        if level is None or not text:
            return
        while self._headings and self._headings[-1][0] >= level:
            self._headings.pop()
        self._headings.append((level, text))
        self.assembler.start_section(" > ".join(title for _, title in self._headings))
        self.assembler.add(text)


def parse_html(data: bytes) -> ParsedDocument:
    extractor = _Extractor()
    extractor.feed(decode_utf8(data))
    extractor.close()
    return extractor.finish()
