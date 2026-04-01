from backend.config import Settings
from backend.models.loader import get_embedding_model, get_nli_model
from backend.core.chunker import Chunker
from backend.core.verifiers.nli import NLIVerifier
from backend.core.verifiers.similarity import SimilarityVerifier
from backend.core.verifiers.consistency import ConsistencyVerifier
from backend.core.ensemble import EnsembleScorer
from backend.core.pipeline import Pipeline
from backend.db.repository import Repository
from backend.api.app import create_app


def create_configured_app():
    """Create the FastAPI app with all real dependencies wired up."""
    settings = Settings()
    app = create_app()

    @app.on_event("startup")
    async def startup():
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
        )

        repo = Repository(settings.db_path)
        await repo.init()

        app.state.pipeline = pipeline
        app.state.repo = repo

    @app.on_event("shutdown")
    async def shutdown():
        if hasattr(app.state, "repo"):
            await app.state.repo.close()

    return app
