from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Models
    nli_model_name: str = "cross-encoder/nli-deberta-v3-large"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_chunks: int = 5

    # Self-consistency
    consistency_samples: int = 5
    consistency_temperature: float = 0.7
    consistency_similarity_threshold: float = 0.85

    # Ensemble weights
    nli_weight: float = 0.5
    consistency_weight: float = 0.3
    similarity_weight: float = 0.2

    # Database
    db_path: str = "halluciscope.db"

    # Document index cache
    index_cache_maxsize: int = 32

    model_config = {"env_prefix": "HALLUCISCOPE_"}
