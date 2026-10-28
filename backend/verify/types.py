"""Value types shared by claim extraction and verification."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


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
