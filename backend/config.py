from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # EvidenceGraph LLM client (answer generation and claim extraction)
    llm_model: str = "llama3.2:3b"
    llm_timeout_s: float = 120.0

    # EvidenceGraph retrieval models (the embedding column is vector(384))
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model_device: str = "auto"
    embed_batch_size: int = 32

    # EvidenceGraph NLI scorer
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    nli_batch_size: int = 16

    # EvidenceGraph calibrator artifact (the worker refuses analyses when it is missing)
    calibrator_path: str = "backend/calibration/artifacts/default.json"

    # EvidenceGraph analyses: the run limit, and how many one user may have queued or running
    analysis_timeout_s: float = 300.0
    max_active_analyses: int = 3

    # Quick checks: how long the deprecated POST /api/analyze waits for the worker
    quick_wait_s: float = 120.0

    # Ingestion limits
    max_upload_bytes: int = 20_971_520
    parse_timeout_s: float = 60.0

    # Search: how long POST /search waits for a reranked result computed by the worker
    search_wait_s: float = 15.0

    # Self-consistency diagnostic of ask analyses: 0 means off (a request sets its own
    # options.consistency_samples)
    consistency_samples: int = 0

    # Postgres
    database_url: str = "postgresql://eg_app@localhost:54329/evidencegraph"
    migration_database_url: str = "postgresql://eg_owner@localhost:54329/evidencegraph"

    # Web / auth
    cors_origins: list[str] = ["http://localhost:5173"]
    allow_registration: bool = True
    dev_insecure_cookies: bool = False
    metrics_token: str | None = None
    session_ttl_days: int = 7

    # Worker / logging
    worker_concurrency: int = 1
    worker_metrics_port: int = 9101
    log_level: str = "INFO"

    model_config = {"env_prefix": "HALLUCISCOPE_"}
