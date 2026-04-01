import pytest
from unittest.mock import AsyncMock, patch
from backend.core.decomposer import decompose_claims, parse_claims_response


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
    with patch("backend.core.decomposer.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = llm_response
        claims = await decompose_claims(
            answer="Some long answer with multiple facts.",
            base_url="http://localhost:11434",
            model="llama3.1:8b",
        )
        assert len(claims) == 3
        assert claims[0] == "Claim one."
