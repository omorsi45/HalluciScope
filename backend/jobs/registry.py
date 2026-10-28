"""Job handler registry: maps a job kind to the coroutine that runs it."""

from typing import Any

from backend.analysis.handlers import fail_run_analysis, handle_run_analysis
from backend.ingest.handlers import fail_ingest_upload, handle_ingest_upload
from backend.jobs.worker import FailureHandler, Handler
from backend.retrieval.handlers import fail_search, handle_search


def build_handlers(services: Any) -> dict[str, Handler]:
    """Plan C also registers reverify_document and reverify_analysis."""
    return {
        "ingest_upload": handle_ingest_upload,
        "search": handle_search,
        "run_analysis": handle_run_analysis,
    }


def build_failure_handlers(services: Any) -> dict[str, FailureHandler]:
    """Per-kind hooks that settle domain rows (an upload, a search request, an analysis)."""
    return {
        "ingest_upload": fail_ingest_upload,
        "search": fail_search,
        "run_analysis": fail_run_analysis,
    }
