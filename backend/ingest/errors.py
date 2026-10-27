"""Stable error codes for ingestion failures."""

INGEST_ERROR_CODES = frozenset(
    {
        "unsupported_type",
        "bad_encoding",
        "bad_pdf",
        "encrypted_pdf",
        "no_text",
        "too_many_rows",
        "bad_csv",
        "too_large",
        "too_long",
        "too_many_passages",
        "parse_timeout",
        "effective_from_out_of_order",
        "document_deleted",  # upload-only
    }
)


class IngestError(Exception):
    """An ingestion failure carrying a stable, storable error code."""

    def __init__(self, code: str) -> None:
        if code not in INGEST_ERROR_CODES:
            raise ValueError("unknown ingest error code")
        super().__init__(code)
        self.code = code
