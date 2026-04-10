import numpy as np
import pytest
from unittest.mock import MagicMock
from backend.core.chunker import Chunker, DocumentIndex


def _make_mock_embedder():
    """Mock embedding model that returns deterministic normalized vectors."""
    mock = MagicMock()

    def encode(texts, **kwargs):
        vectors = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors)

    mock.encode = encode
    return mock


def test_build_index_returns_document_index(sample_document_text):
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    doc_index = chunker.build_index(sample_document_text)

    assert isinstance(doc_index, DocumentIndex)
    assert len(doc_index.chunks) > 1
    assert doc_index.embeddings.shape[0] == len(doc_index.chunks)
    assert doc_index.faiss_index is not None


def test_build_index_is_stateless(sample_document_text):
    """Two calls with different docs produce independent indexes."""
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    idx1 = chunker.build_index(sample_document_text)
    idx2 = chunker.build_index("Completely different text about quantum physics.")

    assert idx1.chunks != idx2.chunks


def test_retrieve_top_k(sample_document_text):
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    doc_index = chunker.build_index(sample_document_text)
    results = chunker.retrieve("When was Einstein born?", doc_index, top_k=3)

    assert len(results) <= 3
    assert all(isinstance(r, str) for r in results)


def test_retrieve_respects_top_k(sample_document_text):
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    doc_index = chunker.build_index(sample_document_text)
    results = chunker.retrieve("Einstein", doc_index, top_k=1)

    assert len(results) == 1


def test_concurrent_indexes_do_not_interfere(sample_document_text):
    """Two DocumentIndex objects from same Chunker are fully independent."""
    chunker = Chunker(embedding_model=_make_mock_embedder(), chunk_size=100, chunk_overlap=20)
    idx1 = chunker.build_index(sample_document_text)
    idx2 = chunker.build_index("A document about machine learning.")

    r1 = chunker.retrieve("Einstein Nobel Prize", idx1, top_k=2)
    r2 = chunker.retrieve("neural networks", idx2, top_k=2)

    for chunk in r1:
        assert chunk in idx1.chunks
    for chunk in r2:
        assert chunk in idx2.chunks
