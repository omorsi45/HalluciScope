"""Value types shared by claim extraction and verification."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


# Probability arrays use this column order.
VERDICT_ORDER = (Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT)


@dataclass(frozen=True)
class Claim:
    ordinal: int  # 0-based position among the claims of one answer
    text: str  # citation markers removed
    answer_start: int  # span of the source sentence in the original answer
    answer_end: int
    method: Literal["llm_atomic", "sentence", "sentence_fallback"]
    cited_numbers: tuple[int, ...] = ()  # P numbers cited in the source sentence


@dataclass(frozen=True)
class NLIResult:
    """Class probabilities for one (premise, hypothesis) pair; they sum to 1."""

    entail: float
    contra: float
    neutral: float


@dataclass(frozen=True)
class EvidenceSignal:
    """One retrieved passage scored against a claim."""

    passage_id: str
    text: str
    nli: NLIResult
    rerank: float | None  # cross-encoder score; None when the passage was not reranked
    dense_cos: float  # cosine similarity between claim and passage embeddings


class NLIScorer(Protocol):
    model_id: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        """Score each (premise, hypothesis) pair, in input order; no pairs gives `[]`."""
        ...
