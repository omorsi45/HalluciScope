import numpy as np
from unittest.mock import MagicMock
from backend.core.chunker import Chunker


def _make_mock_embedder():
    """Mock embedding model that returns deterministic vectors."""
    mock = MagicMock()

    def encode(texts, **kwargs):
        vectors = []
        for i, t in enumerate(texts):
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vectors.append(rng.randn(384).astype(np.float32))
        return np.array(vectors)

    mock.encode = encode
    return mock


def test_chunk_text(sample_document_text):
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    chunker.index_document(sample_document_text)
    assert len(chunker.chunks) > 1
    for chunk in chunker.chunks:
        assert len(chunk.strip()) > 0


def test_retrieve_top_k(sample_document_text):
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    chunker.index_document(sample_document_text)
    results = chunker.retrieve("When was Einstein born?", top_k=3)
    assert len(results) <= 3
    assert all(isinstance(r, str) for r in results)


def test_retrieve_respects_top_k(sample_document_text):
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    chunker.index_document(sample_document_text)
    results = chunker.retrieve("Einstein", top_k=1)
    assert len(results) == 1
