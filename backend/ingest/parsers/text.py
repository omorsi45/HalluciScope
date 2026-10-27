"""Plain text parser: UTF-8 decoding and paragraph splitting."""

import re

from backend.ingest.assemble import TextAssembler
from backend.ingest.errors import IngestError
from backend.ingest.types import ParsedDocument

_BLANK_LINE = re.compile(r"\n\s*\n")


def decode_utf8(data: bytes) -> str:
    """Decode strict UTF-8, dropping a leading byte order mark."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # `from None` keeps the offending bytes out of any traceback.
        raise IngestError("bad_encoding") from None


def parse_text(data: bytes) -> ParsedDocument:
    assembler = TextAssembler()
    for paragraph in _BLANK_LINE.split(decode_utf8(data)):
        assembler.add(paragraph)
    return assembler.build({}, "text", "text/1")
