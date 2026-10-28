"""Job handler registry: maps a job kind to the coroutine that runs it."""

from typing import Any

from backend.ingest.handlers import fail_ingest_upload, handle_ingest_upload
from backend.jobs.worker import FailureHandler, Handler


def build_handlers(services: Any) -> dict[str, Handler]:
    """Plans B and C register ingest_upload, search, run_analysis and reverify_document."""
    return {"ingest_upload": handle_ingest_upload}


def build_failure_handlers(services: Any) -> dict[str, FailureHandler]:
    """Per-kind hooks that settle domain rows (for example an upload) when a job dies."""
    return {"ingest_upload": fail_ingest_upload}
