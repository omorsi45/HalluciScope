"""Job handler registry: maps a job kind to the coroutine that runs it."""

from typing import Any

from backend.ingest.handlers import fail_ingest_upload, handle_ingest_upload
from backend.jobs.worker import FailureHandler, Handler
from backend.retrieval.handlers import fail_search, handle_search


def build_handlers(services: Any) -> dict[str, Handler]:
    """Plan C registers run_analysis and reverify_document."""
    return {"ingest_upload": handle_ingest_upload, "search": handle_search}


def build_failure_handlers(services: Any) -> dict[str, FailureHandler]:
    """Per-kind hooks that settle domain rows (an upload, a search request) when a job dies."""
    return {"ingest_upload": fail_ingest_upload, "search": fail_search}
