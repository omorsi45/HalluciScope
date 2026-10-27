"""Job handler registry: maps a job kind to the coroutine that runs it."""

from typing import Any

from backend.jobs.worker import FailureHandler, Handler


def build_handlers(services: Any) -> dict[str, Handler]:
    """Plans B and C register ingest_upload, search, run_analysis and reverify_document."""
    return {}


def build_failure_handlers(services: Any) -> dict[str, FailureHandler]:
    """Per-kind hooks that settle domain rows (for example an upload) when a job dies."""
    return {}
