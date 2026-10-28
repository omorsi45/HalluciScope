from collections.abc import Sequence

import numpy as np

from backend.config import Settings
from backend.verify.nli import DebertaNLIScorer

# Model adapters. torch and sentence-transformers are imported inside the functions and classes
# below, so importing this module loads no ML library.

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def resolve_device(setting: str) -> str:
    """Map the `model_device` setting to a torch device name; explicit values pass through."""
    if setting != "auto":
        return setting
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise ValueError("embedding model does not report its dimension")
        self.dim = dim

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self._query_prefix + text for text in texts])


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if len(passages) == 0:
            return []
        import torch

        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            # Identity keeps the raw logits whatever default the checkpoint config carries.
            activation_fn=torch.nn.Identity(),
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


def load_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedder_model,
        device=resolve_device(settings.model_device),
        batch_size=settings.embed_batch_size,
    )


def load_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        settings.reranker_model,
        device=resolve_device(settings.model_device),
        batch_size=settings.embed_batch_size,
    )


def load_nli(settings: Settings) -> DebertaNLIScorer:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model)
    return DebertaNLIScorer(
        tokenizer,
        model,
        model_id=settings.nli_model,
        device=resolve_device(settings.model_device),
        batch_size=settings.nli_batch_size,
    )
