"""Job handler registry: maps a job kind to the coroutine that runs it."""

from typing import Any

from backend.jobs.worker import Handler


def build_handlers(services: Any) -> dict[str, Handler]:
    """Plans B and C register ingest_upload, search, run_analysis and reverify_document."""
    return {}
