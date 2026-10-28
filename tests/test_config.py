from backend.config import Settings

# Settings of the pipeline that EvidenceGraph replaced. Nothing reads them any more.
LEGACY_SETTINGS = (
    "ollama_model",
    "embedding_model_name",
    "nli_model_name",
    "chunk_size",
    "chunk_overlap",
    "top_k_chunks",
    "consistency_temperature",
    "consistency_similarity_threshold",
    "nli_weight",
    "consistency_weight",
    "similarity_weight",
    "db_path",
    "index_cache_maxsize",
)


def test_default_settings():
    settings = Settings()
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "llama3.2:3b"
    assert settings.nli_model == "cross-encoder/nli-deberta-v3-large"
    assert settings.consistency_samples == 0


def test_legacy_settings_removed():
    assert [name for name in LEGACY_SETTINGS if name in Settings.model_fields] == []
