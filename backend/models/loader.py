from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_nli_model():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model_name)
    model.eval()
    return tokenizer, model
