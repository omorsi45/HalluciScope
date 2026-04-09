import pytest
import hashlib
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.pipeline import Pipeline, AnalysisResult
from backend.core.chunker import DocumentIndex
from backend.core.verifiers.base import ClaimScore
from backend.core.ensemble import ScoredClaim, ConfidenceTier


def _make_settings():
    s = MagicMock()
    s.ollama_base_url = "http://localhost:11434"
    s.ollama_model = "llama3.1:8b"
    s.index_cache_maxsize = 32
    return s


@pytest.fixture
def mock_pipeline():
    return Pipeline(
        settings=_make_settings(),
        chunker=MagicMock(),
        nli_verifier=MagicMock(),
        similarity_verifier=MagicMock(),
        consistency_verifier=MagicMock(),
        ensemble=MagicMock(),
        http_client=None,
    )


@pytest.mark.asyncio
async def test_pipeline_analyze(mock_pipeline):
    mock_doc_index = MagicMock(spec=DocumentIndex)
    mock_pipeline.chunker.build_index = MagicMock(return_value=mock_doc_index)
    mock_pipeline.chunker.retrieve = MagicMock(return_value=["chunk1", "chunk2"])

    with patch("backend.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Einstein was born in 1879."

        with patch("backend.core.pipeline.decompose_claims", new_callable=AsyncMock) as mock_decomp:
            mock_decomp.return_value = ["Einstein was born in 1879."]

            nli_score = ClaimScore(claim="Einstein was born in 1879.", hallucination_score=0.1, details={"verifier": "nli"})
            cons_score = ClaimScore(claim="Einstein was born in 1879.", hallucination_score=0.0, details={"verifier": "consistency"})
            sim_score = ClaimScore(claim="Einstein was born in 1879.", hallucination_score=0.2, details={"verifier": "similarity"})

            mock_pipeline.nli_verifier.verify = AsyncMock(return_value=[nli_score])
            mock_pipeline.similarity_verifier.verify = AsyncMock(return_value=[sim_score])
            mock_pipeline.consistency_verifier.verify = AsyncMock(return_value=[cons_score])

            scored = ScoredClaim(
                claim="Einstein was born in 1879.",
                hallucination_score=0.09,
                tier=ConfidenceTier.SUPPORTED,
                verifier_details={},
            )
            mock_pipeline.ensemble.score = MagicMock(return_value=[scored])

            result = await mock_pipeline.analyze(
                document_text="Albert Einstein was born in 1879.",
                question="When was Einstein born?",
            )

    assert isinstance(result, AnalysisResult)
    assert result.answer == "Einstein was born in 1879."
    assert len(result.scored_claims) == 1
    assert result.scored_claims[0].tier == ConfidenceTier.SUPPORTED
    assert len(result.retrieved_chunks) == 2

    # Verify question is passed to consistency verifier
    mock_pipeline.consistency_verifier.verify.assert_called_once()
    call_kwargs = mock_pipeline.consistency_verifier.verify.call_args
    assert call_kwargs.kwargs.get("question") == "When was Einstein born?"


@pytest.mark.asyncio
async def test_pipeline_overall_score(mock_pipeline):
    mock_pipeline.chunker.build_index = MagicMock(return_value=MagicMock(spec=DocumentIndex))
    mock_pipeline.chunker.retrieve = MagicMock(return_value=["chunk"])

    with patch("backend.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Answer."
        with patch("backend.core.pipeline.decompose_claims", new_callable=AsyncMock) as mock_decomp:
            mock_decomp.return_value = ["Claim A.", "Claim B."]

            nli_scores = [
                ClaimScore(claim="Claim A.", hallucination_score=0.1, details={}),
                ClaimScore(claim="Claim B.", hallucination_score=0.9, details={}),
            ]
            cons_scores = [
                ClaimScore(claim="Claim A.", hallucination_score=0.0, details={}),
                ClaimScore(claim="Claim B.", hallucination_score=0.8, details={}),
            ]
            sim_scores = [
                ClaimScore(claim="Claim A.", hallucination_score=0.1, details={}),
                ClaimScore(claim="Claim B.", hallucination_score=0.7, details={}),
            ]

            mock_pipeline.nli_verifier.verify = AsyncMock(return_value=nli_scores)
            mock_pipeline.similarity_verifier.verify = AsyncMock(return_value=sim_scores)
            mock_pipeline.consistency_verifier.verify = AsyncMock(return_value=cons_scores)

            scored = [
                ScoredClaim(claim="Claim A.", hallucination_score=0.07, tier=ConfidenceTier.SUPPORTED, verifier_details={}),
                ScoredClaim(claim="Claim B.", hallucination_score=0.83, tier=ConfidenceTier.HALLUCINATED, verifier_details={}),
            ]
            mock_pipeline.ensemble.score = MagicMock(return_value=scored)

            result = await mock_pipeline.analyze(document_text="Doc.", question="Q?")
            assert abs(result.overall_score - 0.45) < 1e-6


@pytest.mark.asyncio
async def test_pipeline_caches_document_index(mock_pipeline):
    """Same document text submitted twice reuses the cached DocumentIndex."""
    mock_pipeline.chunker.build_index = MagicMock(return_value=MagicMock(spec=DocumentIndex))
    mock_pipeline.chunker.retrieve = MagicMock(return_value=[])

    with patch("backend.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Answer."
        with patch("backend.core.pipeline.decompose_claims", new_callable=AsyncMock) as mock_decomp:
            mock_decomp.return_value = []

            await mock_pipeline.analyze(document_text="Same document.", question="Q1?")
            await mock_pipeline.analyze(document_text="Same document.", question="Q2?")

    assert mock_pipeline.chunker.build_index.call_count == 1


@pytest.mark.asyncio
async def test_pipeline_cache_evicts_oldest(mock_pipeline):
    """When cache is full, the oldest entry is evicted."""
    mock_pipeline._cache_maxsize = 2
    mock_pipeline.chunker.build_index = MagicMock(return_value=MagicMock(spec=DocumentIndex))
    mock_pipeline.chunker.retrieve = MagicMock(return_value=[])

    with patch("backend.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Answer."
        with patch("backend.core.pipeline.decompose_claims", new_callable=AsyncMock) as mock_decomp:
            mock_decomp.return_value = []

            await mock_pipeline.analyze(document_text="Doc A.", question="Q?")
            await mock_pipeline.analyze(document_text="Doc B.", question="Q?")
            await mock_pipeline.analyze(document_text="Doc C.", question="Q?")

    assert len(mock_pipeline._index_cache) == 2
    hash_a = hashlib.sha256("Doc A.".encode()).hexdigest()
    assert hash_a not in mock_pipeline._index_cache
