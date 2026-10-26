import re

from backend.core.generator import chat_completion

DECOMPOSE_SYSTEM = """You are a claim decomposition system. Your job is to \
split an answer into independent, verifiable factual statements.

Rules:
- Each claim must be a single, standalone factual statement
- Remove hedging language (probably, might, could) and tag it separately
- Split compound sentences into individual claims
- Skip subjective opinions or vague statements
- Number each claim

Example:
Answer: "Einstein was probably born in 1879 in Ulm and he developed relativity."
Claims:
1. Einstein was born in 1879.
2. Einstein was born in Ulm.
3. Einstein developed relativity."""

DECOMPOSE_USER = """Split this answer into independent, verifiable factual claims. \
Number each claim.

Answer: {answer}

Claims:"""


def parse_claims_response(response: str) -> list[str]:
    """Parse numbered or bulleted claims from an LLM response."""
    if not response.strip():
        return []

    lines = response.strip().split("\n")
    claims = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-\*]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            claims.append(cleaned)
    return claims


async def decompose_claims(
    answer: str,
    base_url: str,
    model: str,
    client=None,
) -> list[str]:
    """Decompose an answer into atomic verifiable claims using the LLM."""
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": DECOMPOSE_USER.format(answer=answer)},
    ]
    raw = await chat_completion(messages, base_url, model, 0.0, client)
    return parse_claims_response(raw)
