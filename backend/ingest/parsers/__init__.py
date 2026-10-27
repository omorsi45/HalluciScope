"""Parser dispatch by file extension, plus a killable out-of-process variant.

Uploads are untrusted, so the worker calls `parse_with_timeout`: the parse runs in a child
process (`backend.ingest.parse_child`) that is killed when it exceeds the time limit.
"""

import asyncio
import json
import os
import sys

from backend.ingest.errors import IngestError
from backend.ingest.parsers.csv_table import parse_csv
from backend.ingest.parsers.html import parse_html
from backend.ingest.parsers.markdown import parse_markdown
from backend.ingest.parsers.pdf import parse_pdf
from backend.ingest.parsers.text import parse_text
from backend.ingest.types import PageSpan, ParsedDocument, SectionSpan

EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
}

MAX_CANONICAL_CHARS = 2_000_000
_PDF_MAGIC = b"%PDF-"
_PARSERS = {
    "application/pdf": parse_pdf,
    "text/plain": parse_text,
    "text/markdown": parse_markdown,
    "text/html": parse_html,
    "text/csv": parse_csv,
}


def _extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def mime_for(filename: str) -> str:
    try:
        return EXTENSIONS[_extension(filename)]
    except KeyError:
        raise IngestError("unsupported_type") from None


def parse(data: bytes, filename: str) -> ParsedDocument:
    mime = mime_for(filename)
    # The extension and the content must agree on "is a PDF"; neither is trusted alone.
    if data.startswith(_PDF_MAGIC) != (mime == "application/pdf"):
        raise IngestError("unsupported_type")
    document = _PARSERS[mime](data)
    if len(document.text) > MAX_CANONICAL_CHARS:
        raise IngestError("too_long")
    if not document.text:
        raise IngestError("no_text")
    return document


async def parse_with_timeout(
    data: bytes, filename: str, timeout_s: float = 60.0
) -> ParsedDocument:
    """Run `parse` in a child process, killing and reaping it if it outlives `timeout_s`."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "backend.ingest.parse_child",
        filename,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(data), timeout_s)
    except TimeoutError:
        raise IngestError("parse_timeout") from None
    finally:
        # Also covers cancellation of the caller: never leave a child or its pipes behind.
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    failure = "bad_pdf" if _extension(filename) == ".pdf" else "unsupported_type"
    if proc.returncode != 0:
        raise IngestError(failure)
    try:
        result = json.loads(stdout)
        if result["ok"] is True:
            return _document_from_json(result["doc"])
        raise IngestError(result["code"])
    except IngestError:
        raise
    except (ValueError, KeyError, TypeError):
        # Unparsable child output, an unknown error code, or a malformed document.
        raise IngestError(failure) from None


def _document_from_json(doc: dict) -> ParsedDocument:
    return ParsedDocument(
        text=doc["text"],
        pages=tuple(PageSpan(page, start, end) for page, start, end in doc["pages"]),
        sections=tuple(SectionSpan(start, end, path) for start, end, path in doc["sections"]),
        metadata=dict(doc["metadata"]),
        parser=doc["parser"],
        parser_version=doc["parser_version"],
    )
