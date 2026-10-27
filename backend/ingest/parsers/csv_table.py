"""CSV parser: turns each data row into one `header: value; header: value` piece.

Uses the standard library `csv.reader`, which is linear time and caps a single field at
`csv.field_size_limit()` characters (an oversized field is a `csv.Error`, reported as
`bad_csv`).
"""

import csv
import io

from backend.ingest.assemble import TextAssembler
from backend.ingest.errors import IngestError
from backend.ingest.parsers.text import decode_utf8
from backend.ingest.types import ParsedDocument

MAX_DATA_ROWS = 50_000


def parse_csv(data: bytes) -> ParsedDocument:
    text = decode_utf8(data)
    assembler = TextAssembler()
    header: list[str] | None = None
    rows = 0
    try:
        # `newline=""` hands line endings to `csv`, which needs them to keep quoted newlines.
        for record in csv.reader(io.StringIO(text, newline=""), dialect="excel"):
            if not record:  # a blank line is not a record
                continue
            if header is None:
                header = [cell.strip() for cell in record]
                continue
            rows += 1
            if rows > MAX_DATA_ROWS:
                raise IngestError("too_many_rows")
            pairs: list[str] = []
            for index, cell in enumerate(record):
                value = cell.strip()
                if value:
                    name = header[index] if index < len(header) else ""
                    pairs.append(f"{name or 'column_' + str(index + 1)}: {value}")
            if pairs:
                assembler.add("; ".join(pairs))
    except csv.Error:
        # `from None` keeps the document text out of any traceback.
        raise IngestError("bad_csv") from None
    document = assembler.build({}, "csv", "csv/1")
    if not document.text:
        raise IngestError("no_text")
    return document
