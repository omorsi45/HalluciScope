from backend.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3.1:8b"
    assert settings.nli_model_name == "cross-encoder/nli-deberta-v3-large"
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.top_k_chunks == 5
    assert settings.consistency_samples == 5
    assert settings.consistency_temperature == 0.7
    assert settings.consistency_similarity_threshold == 0.85
    assert settings.nli_weight == 0.5
    assert settings.consistency_weight == 0.3
    assert settings.similarity_weight == 0.2
    assert settings.db_path == "halluciscope.db"


def test_settings_weights_sum_to_one():
    settings = Settings()
    total = settings.nli_weight + settings.consistency_weight + settings.similarity_weight
    assert abs(total - 1.0) < 1e-6
