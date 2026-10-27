from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from backend.api.app import create_app
from backend.auth.service import AuthService
from backend.config import Settings
from backend.core.chunker import Chunker
from backend.core.ensemble import EnsembleScorer
from backend.core.pipeline import Pipeline
from backend.core.verifiers.consistency import ConsistencyVerifier
from backend.core.verifiers.nli import NLIVerifier
from backend.core.verifiers.similarity import SimilarityVerifier
from backend.db.pool import Database
from backend.db.repository import Repository
from backend.models.loader import get_embedding_model, get_nli_model
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
        repo = None
        try:
            app.state.db = db
            app.state.auth = AuthService(db, settings)
            http_client = httpx.AsyncClient(timeout=120.0)
            embedding_model = get_embedding_model()
            tokenizer, nli_model = get_nli_model()

            chunker = Chunker(
                embedding_model=embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                top_k=settings.top_k_chunks,
            )
            nli_verifier = NLIVerifier(tokenizer=tokenizer, model=nli_model)
            similarity_verifier = SimilarityVerifier(embedding_model=embedding_model)
            consistency_verifier = ConsistencyVerifier(
                embedding_model=embedding_model,
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                n_samples=settings.consistency_samples,
                temperature=settings.consistency_temperature,
                similarity_threshold=settings.consistency_similarity_threshold,
                http_client=http_client,
            )
            ensemble = EnsembleScorer(
                nli_weight=settings.nli_weight,
                consistency_weight=settings.consistency_weight,
                similarity_weight=settings.similarity_weight,
            )
            pipeline = Pipeline(
                settings=settings,
                chunker=chunker,
                nli_verifier=nli_verifier,
                similarity_verifier=similarity_verifier,
                consistency_verifier=consistency_verifier,
                ensemble=ensemble,
                http_client=http_client,
            )
            repo = Repository(settings.db_path)
            await repo.init()

            app.state.pipeline = pipeline
            app.state.repo = repo
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
                try:
                    if repo is not None:
                        await repo.close()
                finally:
                    await db.close()

    return create_app(settings, lifespan=lifespan)
