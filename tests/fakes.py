"""Deterministic stand-ins for the embedding, reranking and NLI models, and a scripted LLM.

The embedding and reranking fakes do not use Python's `hash`, which is randomized per
process, so vectors and scores are identical across runs and across processes.
"""

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from backend.verify.claims import STOPWORDS
from backend.verify.types import NLIResult

_TOKEN = re.compile(r"\w+")
_NUMBER = re.compile(r"\d+")


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


class RuleNLIScorer:
    """NLI from word overlap and numbers, so verification tests need no model.

    Content tokens are the lowercase word tokens (letters, digits, underscore) outside
    `STOPWORDS`; numbers are the tokens made of digits. For a (premise, hypothesis) pair:
    - contradiction: the hypothesis has numbers, none of them occur in the premise, the
      premise has numbers, and every non-number content token of the hypothesis occurs in the
      premise (the same statement with other figures);
    - entailment: every content token of the hypothesis occurs in the premise;
    - neutral: anything else.
    The chosen class gets 0.9 and each of the others 0.05.
    """

    model_id = "fake-rule-nli"

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        return [self._score_pair(premise, hypothesis) for premise, hypothesis in pairs]

    @staticmethod
    def _score_pair(premise: str, hypothesis: str) -> NLIResult:
        premise_tokens = _content_tokens(premise)
        hypothesis_tokens = _content_tokens(hypothesis)
        premise_numbers = {token for token in premise_tokens if _NUMBER.fullmatch(token)}
        hypothesis_numbers = {token for token in hypothesis_tokens if _NUMBER.fullmatch(token)}
        hypothesis_words = hypothesis_tokens - hypothesis_numbers

        if (
            hypothesis_numbers
            and not hypothesis_numbers & premise_numbers
            and premise_numbers
            and hypothesis_words <= premise_tokens
        ):
            return NLIResult(entail=0.05, contra=0.9, neutral=0.05)
        if hypothesis_tokens <= premise_tokens:
            return NLIResult(entail=0.9, contra=0.05, neutral=0.05)
        return NLIResult(entail=0.05, contra=0.05, neutral=0.9)


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in STOPWORDS}


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
