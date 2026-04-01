import pytest
from unittest.mock import AsyncMock, patch
from backend.core.generator import generate_answer


@pytest.mark.asyncio
async def test_generate_answer():
    mock_response = {
        "message": {"content": "Einstein was born in 1879 in Ulm, Germany."}
    }
    with patch("backend.core.generator.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = AsyncMock(
            status_code=200,
            json=lambda: mock_response,
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        result = await generate_answer(
            question="When was Einstein born?",
            context_chunks=["Albert Einstein was born on March 14, 1879, in Ulm, Germany."],
            base_url="http://localhost:11434",
            model="llama3.1:8b",
        )
        assert "Einstein" in result or "1879" in result

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        prompt_content = str(body["messages"])
        assert "Ulm" in prompt_content


@pytest.mark.asyncio
async def test_generate_answer_with_temperature():
    mock_response = {
        "message": {"content": "Some answer."}
    }
    with patch("backend.core.generator.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = AsyncMock(
            status_code=200,
            json=lambda: mock_response,
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        result = await generate_answer(
            question="test",
            context_chunks=["context"],
            base_url="http://localhost:11434",
            model="llama3.1:8b",
            temperature=0.7,
        )
        assert result == "Some answer."

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["options"]["temperature"] == 0.7
