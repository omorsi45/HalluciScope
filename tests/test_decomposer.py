import pytest
from unittest.mock import AsyncMock, patch
from backend.core.decomposer import DECOMPOSE_SYSTEM, decompose_claims, parse_claims_response


def test_parse_claims_response_numbered():
    response = """1. Einstein was born in 1879.
2. He was born in Ulm, Germany.
3. He developed special relativity."""
    claims = parse_claims_response(response)
    assert len(claims) == 3
    assert claims[0] == "Einstein was born in 1879."
    assert claims[1] == "He was born in Ulm, Germany."
    assert claims[2] == "He developed special relativity."


def test_parse_claims_response_dashed():
    response = """- Einstein was born in 1879.
- He was born in Ulm, Germany."""
    claims = parse_claims_response(response)
    assert len(claims) == 2


def test_parse_claims_response_strips_whitespace():
    response = """1.   Einstein was born in 1879.
2.  He was born in Ulm.  """
    claims = parse_claims_response(response)
    assert claims[0] == "Einstein was born in 1879."
    assert claims[1] == "He was born in Ulm."


def test_parse_claims_empty():
    assert parse_claims_response("") == []
    assert parse_claims_response("   ") == []


@pytest.mark.asyncio
async def test_decompose_claims():
    llm_response = "1. Claim one.\n2. Claim two.\n3. Claim three."
    with patch("backend.core.decomposer.chat_completion", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = llm_response
        claims = await decompose_claims(
            answer="Some long answer with multiple facts.",
            base_url="http://localhost:11434",
            model="llama3.1:8b",
        )
        assert len(claims) == 3
        assert claims[0] == "Claim one."


@pytest.mark.asyncio
async def test_decompose_uses_decompose_system_prompt():
    answer = "Einstein was born in Ulm and developed relativity."
    with patch("backend.core.decomposer.chat_completion", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "1. A.\n2. B."
        claims = await decompose_claims(
            answer=answer,
            base_url="http://localhost:11434",
            model="llama3.1:8b",
        )
    messages = mock_chat.call_args.args[0]
    assert messages[0] == {"role": "system", "content": DECOMPOSE_SYSTEM}
    assert "Context:" not in messages[1]["content"]
    assert answer in messages[1]["content"]
    assert claims == ["A.", "B."]
