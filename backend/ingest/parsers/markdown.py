"""Markdown parser: strips syntax to readable text and records heading sections.

Line based and original (no Markdown library). Every piece goes through `TextAssembler`, so
normalization and exact offsets come from it.

Uploads are untrusted and a regex cannot be interrupted by the parse timeout, so every
pattern here either is anchored to the start of a line, has a bounded repeat (`_MAX_SPAN`),
or is only ever searched forward from a position that moves past each match.
"""

import re

from backend.ingest.assemble import TextAssembler
from backend.ingest.normalize import normalize
from backend.ingest.parsers.text import decode_utf8
from backend.ingest.types import ParsedDocument

# Longest inline span (emphasis, code, link text) that is recognised. Longer spans keep their
# markers; the bound keeps the work per marker constant on hostile lines.
_MAX_SPAN = 300

_NEWLINE = re.compile(r"\r\n|\r|\n")
# A backtick fence cannot carry backticks in its info string, or it is an inline code span.
_FENCE_OPEN = re.compile(r"^\s*(`{3,}(?=[^`]*$)|~{3,})")
_FENCE_CLOSE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")
# ATX heading: the closing `\s*#*\s*` of the usual pattern is trimmed with rstrip instead,
# because that lazy form backtracks quadratically on long whitespace runs.
_HEADING = re.compile(r"(#{1,6})\s+(.*)")
_BLOCKQUOTE = re.compile(r"^ {0,3}(?:>\s?)+")
_HRULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$")
_LIST_MARKER = re.compile(r"^(\s*)[*+](?=\s)")

# Starts at the beginning of a backtick run and takes the whole run (possessive), so a long
# run of backticks is tried once rather than once per length.
_CODE_SPAN = re.compile(rf"(?<!`)(?P<ticks>`++)(?P<code>.{{1,{_MAX_SPAN}}}?)(?<!`)(?P=ticks)(?!`)")
_LINK_START = re.compile(rf"!?\[([^\]]{{0,{_MAX_SPAN}}})\]\(")
# Markers must hug non-space text; underscore markers must not sit inside a word. The closing
# marker is matched before its lookbehind so the lazy scan stays cheap.
_EMPHASIS = (
    re.compile(rf"\*\*(?=\S)(.{{1,{_MAX_SPAN}}}?)\*\*(?<=\S\*\*)"),
    re.compile(rf"(?<!\w)__(?=\S)(.{{1,{_MAX_SPAN}}}?)__(?<=\S__)(?!\w)"),
    re.compile(rf"\*(?=\S)(.{{1,{_MAX_SPAN}}}?)\*(?<=\S\*)"),
    re.compile(rf"(?<!\w)_(?=\S)(.{{1,{_MAX_SPAN}}}?)_(?<=\S_)(?!\w)"),
)
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def parse_markdown(data: bytes) -> ParsedDocument:
    assembler = TextAssembler()
    metadata: dict[str, str] = {}
    headings: list[tuple[int, str]] = []
    paragraph: list[str] = []
    code: list[str] = []
    fence: str | None = None
    in_comment = False

    def flush() -> None:
        if paragraph:
            assembler.add("\n".join(paragraph))
            paragraph.clear()

    for raw in _NEWLINE.split(decode_utf8(data)):
        if fence is not None:
            closing = _FENCE_CLOSE.match(raw)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence):
                assembler.add("\n".join(code))
                code.clear()
                fence = None
            else:
                code.append(raw)
            continue

        if not in_comment:
            opening = _FENCE_OPEN.match(raw)
            if opening:
                flush()
                fence = opening.group(1)
                continue

        line, in_comment = _strip_comments(raw, in_comment)
        line = _BLOCKQUOTE.sub("", line)
        if not line.strip():
            flush()
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            text = normalize(_clean_inline(heading.group(2).rstrip().rstrip("#")))
            if text:
                level = len(heading.group(1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, text))
                assembler.start_section(" > ".join(title for _, title in headings))
                assembler.add(text)
                if level == 1 and "title" not in metadata:
                    metadata["title"] = text
            continue

        if _HRULE.match(line) or _REF_DEF.match(line):
            flush()
            continue

        if line.lstrip().startswith("|"):
            row = _table_row(line)
            if row is None:
                continue
            line = row
        else:
            line = _clean_inline(_LIST_MARKER.sub(r"\1-", line))
            if not line.strip():
                flush()
                continue
        paragraph.append(line)

    flush()
    if fence is not None:
        # An unclosed fence runs to the end of the document.
        assembler.add("\n".join(code))
    return assembler.build(metadata, "markdown", "markdown/1")


def _strip_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """Remove `<!-- ... -->` text from one line; the flag carries a comment across lines."""
    kept: list[str] = []
    pos = 0
    while pos < len(line):
        if in_comment:
            end = line.find("-->", pos)
            if end < 0:
                break
            pos = end + 3
            in_comment = False
        else:
            start = line.find("<!--", pos)
            if start < 0:
                kept.append(line[pos:])
                break
            kept.append(line[pos:start])
            pos = start + 4
            in_comment = True
    return "".join(kept), in_comment


def _table_row(line: str) -> str | None:
    """Join the non-empty cells of a pipe table row; separator and empty rows give None."""
    row = line.strip()
    if _TABLE_SEPARATOR.match(row):
        return None
    cells = [_clean_inline(cell.strip()) for cell in row.strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    return "; ".join(cells) if cells else None


def _strip_links(text: str) -> str:
    """Replace `![alt](url)` and `[text](url)` with their label.

    The URL ends at the first `)` and is found with `str.find`, so it can be any length (an
    inline data URI) at linear cost. With no `)` left, no later link can close either.
    """
    parts: list[str] = []
    pos = 0
    while True:
        start = _LINK_START.search(text, pos)
        if start is None:
            break
        close = text.find(")", start.end())
        if close < 0:
            break
        parts.append(text[pos : start.start()])
        parts.append(start.group(1))
        pos = close + 1
    parts.append(text[pos:])
    return "".join(parts)


def _clean_inline(text: str) -> str:
    """Strip inline Markdown from one line, keeping the readable content."""
    # NUL cannot occur in canonical text, so it is a safe placeholder delimiter once removed.
    text = text.replace("\x00", "")
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group("code"))
        return f"\x00{len(spans) - 1}\x00"

    # Code spans are lifted out first so their content is never treated as syntax.
    text = _strip_links(_CODE_SPAN.sub(stash, text))
    for pattern in _EMPHASIS:
        text = pattern.sub(r"\1", text)
    return _PLACEHOLDER.sub(lambda match: spans[int(match.group(1))], text)
