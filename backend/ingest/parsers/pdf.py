"""PDF parser built on pypdf: one canonical-text piece per page, with page spans."""

import io

from pypdf import PasswordType, PdfReader
from pypdf.errors import PyPdfError

from backend.ingest.assemble import TextAssembler
from backend.ingest.errors import IngestError
from backend.ingest.normalize import normalize
from backend.ingest.types import ParsedDocument


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        return _parse(data)
    except (PyPdfError, ValueError, KeyError, TypeError):
        # `from None` keeps the document content out of any traceback.
        raise IngestError("bad_pdf") from None


def _parse(data: bytes) -> ParsedDocument:
    reader = PdfReader(io.BytesIO(data))
    # Many PDFs are "encrypted" with an empty user password only to restrict printing.
    if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
        raise IngestError("encrypted_pdf")
    assembler = TextAssembler()
    for number, page in enumerate(reader.pages, start=1):
        assembler.add(page.extract_text() or "", page=number)
    document = assembler.build(_metadata(reader), "pdf", "pdf/1")
    if not document.text:
        raise IngestError("no_text")
    return document


def _metadata(reader: PdfReader) -> dict[str, str]:
    metadata: dict[str, str] = {}
    info = reader.metadata
    if info is None:
        return metadata
    for key, value in (("title", info.title), ("author", info.author)):
        if isinstance(value, str) and (clean := normalize(value)):
            metadata[key] = clean
    try:
        created = info.creation_date
    except ValueError:
        # An unreadable date is missing metadata, not a corrupt file.
        created = None
    if created is not None:
        metadata["created"] = created.isoformat()
    return metadata
