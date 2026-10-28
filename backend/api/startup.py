from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from backend.api.app import create_app
from backend.auth.service import AuthService
from backend.config import Settings
from backend.db.pool import Database
from backend.models.loader import load_embedder
from backend.obs.logging import configure_logging


def create_configured_app() -> FastAPI:
    """Create the FastAPI app with all real dependencies wired up.

    The lifespan calls ``configure_logging`` after uvicorn has applied its own logging
    config. ``configure_logging`` replaces the root handlers with one JSON handler, points
    the ``uvicorn`` logger (and so ``uvicorn.error``, which propagates to it) at that handler
    with propagation off, and empties, un-propagates and disables ``uvicorn.access`` because
    the request middleware already logs each request. Nothing else is required: run uvicorn
    normally. Do not pass ``--log-config /dev/null`` on the CLI (it is not a valid config
    file); programmatic ``uvicorn.run(..., log_config=None)`` is also fine.
    """
    settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── startup ──────────────────────────────────────────────────
        configure_logging(settings.log_level)
        db = Database(settings.database_url)
        await db.open()
        http_client = None
        try:
            app.state.db = db
            app.state.auth = AuthService(db, settings)
            # R7: the API process loads only the embedder; the reranker, the NLI model and the
            # LLM client live in the worker, which runs the analyses.
            app.state.embedder = load_embedder(settings)
            http_client = httpx.AsyncClient(timeout=120.0)
            app.state.http_client = http_client
            app.state.http = http_client

            yield  # ── app runs ─────────────────────────────────────────
        finally:
            # ── shutdown (also reached when startup fails) ───────────
            # Each resource is closed in its own try/finally so one failure cannot leak
            # the others.
            try:
                if http_client is not None:
                    await http_client.aclose()
            finally:
                await db.close()

    return create_app(settings, lifespan=lifespan)
