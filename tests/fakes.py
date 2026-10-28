"""Deterministic stand-ins for the embedding and reranking models, and a scripted LLM.

The embedding and reranking fakes do not use Python's `hash`, which is randomized per
process, so vectors and scores are identical across runs and across processes.
"""

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashEmbedder:
    """Bag-of-words feature hashing: texts sharing words get a higher cosine."""

    model_id = "fake-hash-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            # No tokens (or every token cancelled out): use one fixed unit vector.
            vector[0] = 1.0
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        # Queries and passages share one embedding so that shared words raise the cosine.
        return self.embed_passages(texts)


class KeywordReranker:
    """Scores a passage by how many distinct query tokens it contains."""

    model_id = "fake-keyword"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(_tokens(query))
        return [float(len(query_tokens & set(_tokens(passage)))) for passage in passages]



class ScriptedLLM:
    """LLM client fake: replies come from `responses` in order, then from `default`.

    A response that is an Exception is raised instead of returned, so tests can script
    failures. Every call is recorded in `calls`, including the ones that raise.
    """

    def __init__(self, model_id: str = "fake-llm") -> None:
        self.model_id = model_id
        self.responses: list[str | Exception] = []
        self.default: Callable[[Sequence[Mapping[str, str]]], str] = lambda messages: ""
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        json_schema: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "temperature": temperature,
                "json_schema": json_schema,
            }
        )
        if not self.responses:
            return self.default(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
