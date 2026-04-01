import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.core.verifiers.similarity import SimilarityVerifier


def _make_mock_embedder():
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


def _make_identity_embedder():
    mock = MagicMock()
    def encode(texts, **kwargs):
        vec = np.ones(384, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return np.array([vec] * len(texts))
    mock.encode = encode
    return mock


@pytest.mark.asyncio
async def test_similarity_identical_texts():
    verifier = SimilarityVerifier(embedding_model=_make_identity_embedder())
    results = await verifier.verify(
        claims=["Einstein was born in 1879."],
        context_chunks=["Einstein was born in 1879."],
    )
    assert len(results) == 1
    assert results[0].hallucination_score < 0.01


@pytest.mark.asyncio
async def test_similarity_returns_correct_count():
    verifier = SimilarityVerifier(embedding_model=_make_mock_embedder())
    results = await verifier.verify(
        claims=["Claim A.", "Claim B.", "Claim C."],
        context_chunks=["Some context."],
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_similarity_details_contain_score():
    verifier = SimilarityVerifier(embedding_model=_make_mock_embedder())
    results = await verifier.verify(
        claims=["A claim."],
        context_chunks=["Context one.", "Context two."],
    )
    assert "max_similarity" in results[0].details
    assert "matched_chunk" in results[0].details
    assert results[0].details["verifier"] == "similarity"
