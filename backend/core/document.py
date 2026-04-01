from pathlib import Path
from pypdf import PdfReader


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
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    # Default: read as plain text
    return file_path.read_text(encoding="utf-8")
