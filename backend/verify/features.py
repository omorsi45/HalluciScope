"""The fixed feature vector the calibrator consumes for one claim.

`FEATURE_NAMES` is part of the calibrator artifact contract: the order and the definitions
below must not change without retraining and re-versioning the artifact.
"""

import math
import re
from collections.abc import Sequence

import numpy as np

from backend.verify.claims import STOPWORDS
from backend.verify.types import EvidenceSignal

FEATURE_NAMES: tuple[str, ...] = (
    "max_entail",
    "max_contra",
    "entail_minus_contra",
    "neutral_at_best",
    "n_entail_ge_half",
    "top_rerank",
    "max_dense_cos",
    "number_coverage",
    "has_numbers",
    "content_coverage",
    "claim_len",
    "contra_and_entail",
)

# Matches are normalized by removing "$" and ",", so "$1,200" and "1200" are the same number
# while "5%" and "5" are not.
NUMBER_RE = re.compile(r"\$?\d+(?:[.,]\d+)*%?")
_WORD = re.compile(r"\w+")


def order_signals(signals: Sequence[EvidenceSignal]) -> list[EvidenceSignal]:
    """Signals by rerank score descending, unscored last; ties keep their input order."""
    return sorted(
        signals,
        key=lambda s: (s.rerank is None, 0.0 if s.rerank is None else -s.rerank),
    )


def compute_features(
    claim_text: str, signals: Sequence[EvidenceSignal], evidence_k: int
) -> np.ndarray:
    """The 12 features of `FEATURE_NAMES` for a claim and the top `evidence_k` signals."""
    used = order_signals(signals)[:evidence_k]
    claim_numbers = _numbers(claim_text)
    claim_words = _WORD.findall(claim_text.lower())
    claim_content = {word for word in claim_words if word not in STOPWORDS}
    claim_len = math.log1p(len(claim_words))
    has_numbers = 1.0 if claim_numbers else 0.0

    if not used:
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0 if claim_numbers else 1.0,
                has_numbers,
                0.0 if claim_content else 1.0,
                claim_len,
                0.0,
            ],
            dtype=np.float64,
        )

    # max() keeps the first of equal values, which is the tie rule for the best signal.
    best = max(used, key=lambda s: s.nli.entail)
    max_entail = best.nli.entail
    max_contra = max(s.nli.contra for s in used)
    reranks = [s.rerank for s in used if s.rerank is not None]

    return np.array(
        [
            max_entail,
            max_contra,
            max_entail - max_contra,
            best.nli.neutral,
            math.log1p(sum(1 for s in used if s.nli.entail >= 0.5)),
            _sigmoid(max(reranks)) if reranks else 0.0,
            max(s.dense_cos for s in used),
            _coverage(claim_numbers, _numbers(best.text)),
            has_numbers,
            _coverage(claim_content, set(_WORD.findall(best.text.lower()))),
            claim_len,
            min(max_entail, max_contra),
        ],
        dtype=np.float64,
    )


def _numbers(text: str) -> set[str]:
    return {match.replace("$", "").replace(",", "") for match in NUMBER_RE.findall(text)}


def _coverage(wanted: set[str], found: set[str]) -> float:
    """Fraction of `wanted` present in `found`; 1.0 when there is nothing to find."""
    if not wanted:
        return 1.0
    return len(wanted & found) / len(wanted)


def _sigmoid(x: float) -> float:
    # Two branches so exp() never sees a large positive argument.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
