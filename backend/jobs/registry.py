"""Job handler registry: maps a job kind to the coroutine that runs it."""

from typing import Any

from backend.analysis.handlers import fail_run_analysis, handle_run_analysis
from backend.ingest.handlers import fail_ingest_upload, handle_ingest_upload
from backend.jobs.worker import FailureHandler, Handler
from backend.provenance.reverify import (
    handle_reverify_analysis,
    handle_reverify_document,
    log_reverify_failure,
)
from backend.retrieval.handlers import fail_search, handle_search


def build_handlers(services: Any) -> dict[str, Handler]:
    return {
        "ingest_upload": handle_ingest_upload,
        "search": handle_search,
        "run_analysis": handle_run_analysis,
        "reverify_document": handle_reverify_document,
        "reverify_analysis": handle_reverify_analysis,
    }


def build_failure_handlers(services: Any) -> dict[str, FailureHandler]:
    """Per-kind hooks that settle domain rows (an upload, a search request, an analysis).

    The re-verification jobs have no row to settle, so their hook only logs.
    """
    return {
        "ingest_upload": fail_ingest_upload,
        "search": fail_search,
        "run_analysis": fail_run_analysis,
        "reverify_document": log_reverify_failure,
        "reverify_analysis": log_reverify_failure,
    }
