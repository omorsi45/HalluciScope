from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from backend.api.app import create_app
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

    The lifespan calls ``configure_logging``, which installs the JSON handler and disables
    ``uvicorn.access`` (the request middleware logs each request instead). uvicorn applies its
    own default ``log_config`` before the lifespan runs and would re-enable ``uvicorn.access``
    on a second configuration pass, so run it with ``--log-config`` pointing at nothing (or
    ``log_config=None`` for programmatic ``uvicorn.run``) to keep a single log format.
    """
    settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── startup ──────────────────────────────────────────────────
        configure_logging(settings.log_level)
        db = Database(settings.database_url)
        await db.open()
        app.state.db = db
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

        yield  # ── app runs ─────────────────────────────────────────

        # ── shutdown ─────────────────────────────────────────────────
        await http_client.aclose()
        await repo.close()
        await db.close()

    return create_app(settings, lifespan=lifespan)
