from pathlib import Path

from backend.ingest.errors import IngestError
from backend.ingest.parsers.pdf import parse_pdf
from backend.ingest.parsers.text import parse_text


def parse_document(
    text: str | None = None,
    file_path: str | Path | None = None,
) -> str:
    """Parse a document from raw text or a file path (PDF or TXT)."""
    if text is not None:
        return text

    if file_path is None:
        raise ValueError("Either text or file_path must be provided.")

    file_path = Path(file_path)
    data = file_path.read_bytes()

    if file_path.suffix.lower() == ".pdf":
        try:
            return parse_pdf(data).text
        except IngestError as exc:
            if exc.code == "no_text":  # an image-only PDF has always read as empty text here
                return ""
            raise

    # Default: read as plain text
    return parse_text(data).text
