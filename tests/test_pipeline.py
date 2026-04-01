import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.pipeline import Pipeline, AnalysisResult
from backend.core.verifiers.base import ClaimScore
from backend.core.ensemble import ScoredClaim, ConfidenceTier


@pytest.fixture
def mock_pipeline():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.chunker = MagicMock()
    pipeline.nli_verifier = MagicMock()
    pipeline.similarity_verifier = MagicMock()
    pipeline.consistency_verifier = MagicMock()
    pipeline.ensemble = MagicMock()
    pipeline.settings = MagicMock()
    pipeline.settings.ollama_base_url = "http://localhost:11434"
    pipeline.settings.ollama_model = "llama3.1:8b"
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_analyze(mock_pipeline):
    mock_pipeline.chunker.index_document = MagicMock()
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


@pytest.mark.asyncio
async def test_pipeline_overall_score(mock_pipeline):
    mock_pipeline.chunker.index_document = MagicMock()
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
