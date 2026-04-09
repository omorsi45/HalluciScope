import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.verifiers.consistency import ConsistencyVerifier


def _make_deterministic_embedder():
    mock = MagicMock()
    cache = {}

    def encode(texts, **kwargs):
        vectors = []
        for t in texts:
            if t not in cache:
                seed = hash(t) % (2**31)
                rng = np.random.RandomState(seed)
                vec = rng.randn(384).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                cache[t] = vec
            vectors.append(cache[t])
        return np.array(vectors)

    mock.encode = encode
    return mock


@pytest.mark.asyncio
async def test_consistency_all_samples_agree():
    embedder = _make_deterministic_embedder()

    async def mock_generate(**kwargs):
        return "Einstein was born in 1879."

    async def mock_decompose(**kwargs):
        return ["Einstein was born in 1879."]

    verifier = ConsistencyVerifier(
        embedding_model=embedder,
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        n_samples=5,
        temperature=0.7,
        similarity_threshold=0.85,
    )

    with patch("backend.core.verifiers.consistency.generate_answer", side_effect=mock_generate):
        with patch("backend.core.verifiers.consistency.decompose_claims", side_effect=mock_decompose):
            results = await verifier.verify(
                claims=["Einstein was born in 1879."],
                context_chunks=["Albert Einstein was born in 1879."],
                question="When was Einstein born?",
            )

    assert len(results) == 1
    assert results[0].hallucination_score == 0.0


@pytest.mark.asyncio
async def test_consistency_no_samples_agree():
    embedder = _make_deterministic_embedder()
    call_count = 0

    async def mock_generate(**kwargs):
        return "Something completely different."

    async def mock_decompose(**kwargs):
        nonlocal call_count
        call_count += 1
        return [f"Unique claim number {call_count}."]

    verifier = ConsistencyVerifier(
        embedding_model=embedder,
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        n_samples=5,
        temperature=0.7,
        similarity_threshold=0.85,
    )

    with patch("backend.core.verifiers.consistency.generate_answer", side_effect=mock_generate):
        with patch("backend.core.verifiers.consistency.decompose_claims", side_effect=mock_decompose):
            results = await verifier.verify(
                claims=["Einstein was born in 1879."],
                context_chunks=["Context."],
                question="When was Einstein born?",
            )

    assert len(results) == 1
    assert results[0].hallucination_score == 1.0


@pytest.mark.asyncio
async def test_consistency_details():
    embedder = _make_deterministic_embedder()

    async def mock_generate(**kwargs):
        return "Answer."

    async def mock_decompose(**kwargs):
        return ["Einstein was born in 1879."]

    verifier = ConsistencyVerifier(
        embedding_model=embedder,
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        n_samples=3,
        temperature=0.7,
        similarity_threshold=0.85,
    )

    with patch("backend.core.verifiers.consistency.generate_answer", side_effect=mock_generate):
        with patch("backend.core.verifiers.consistency.decompose_claims", side_effect=mock_decompose):
            results = await verifier.verify(
                claims=["Einstein was born in 1879."],
                context_chunks=["Context."],
                question="When was Einstein born?",
            )

    assert results[0].details["verifier"] == "consistency"
    assert results[0].details["n_samples"] == 3
    assert "appearances" in results[0].details


@pytest.mark.asyncio
async def test_consistency_uses_real_question():
    """Verify the actual question is passed to _sample_once, not the joined chunks."""
    embedder = _make_deterministic_embedder()
    captured_questions = []

    async def mock_generate(question, **kwargs):
        captured_questions.append(question)
        return "Einstein was born in 1879."

    async def mock_decompose(**kwargs):
        return ["Einstein was born in 1879."]

    verifier = ConsistencyVerifier(
        embedding_model=embedder,
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        n_samples=2,
        temperature=0.7,
        similarity_threshold=0.85,
    )

    with patch("backend.core.verifiers.consistency.generate_answer", side_effect=mock_generate):
        with patch("backend.core.verifiers.consistency.decompose_claims", side_effect=mock_decompose):
            await verifier.verify(
                claims=["Einstein was born in 1879."],
                context_chunks=["Albert Einstein was born in 1879."],
                question="When was Einstein born?",
            )

    assert all(q == "When was Einstein born?" for q in captured_questions)
