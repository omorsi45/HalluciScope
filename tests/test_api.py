import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from backend.api.app import create_app
from backend.core.pipeline import AnalysisResult
from backend.core.ensemble import ScoredClaim, ConfidenceTier


@pytest.fixture
def mock_app():
    return create_app()


@pytest.mark.asyncio
async def test_health(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_analyze_endpoint(mock_app):
    mock_result = AnalysisResult(
        question="When was Einstein born?",
        answer="Einstein was born in 1879.",
        scored_claims=[
            ScoredClaim(
                claim="Einstein was born in 1879.",
                hallucination_score=0.1,
                tier=ConfidenceTier.SUPPORTED,
                verifier_details={"nli": {}, "consistency": {}, "similarity": {}},
            )
        ],
        retrieved_chunks=["Einstein was born on March 14, 1879."],
        overall_score=0.1,
    )

    with patch.object(mock_app.state, "pipeline", create=True) as mock_pipeline:
        mock_pipeline.analyze = AsyncMock(return_value=mock_result)

        with patch.object(mock_app.state, "repo", create=True) as mock_repo:
            mock_repo.save_analysis = AsyncMock(return_value=1)

            transport = ASGITransport(app=mock_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/analyze", json={
                    "document_text": "Einstein was born in 1879.",
                    "question": "When was Einstein born?",
                })

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Einstein was born in 1879."
            assert len(data["scored_claims"]) == 1
            assert data["scored_claims"][0]["tier"] == "supported"
            assert data["overall_score"] == 0.1


@pytest.mark.asyncio
async def test_feedback_endpoint(mock_app):
    with patch.object(mock_app.state, "repo", create=True) as mock_repo:
        mock_repo.save_feedback = AsyncMock(return_value=1)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/feedback", json={
                "analysis_id": 1,
                "claim_index": 0,
                "is_correct": False,
                "note": "Actually true",
            })

        assert response.status_code == 200
        assert response.json()["id"] == 1
